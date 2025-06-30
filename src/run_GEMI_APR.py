import numpy as np
import torch
from genomicWSIdataset import GenomicWSIDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import OrderedDict
from utils import check_dir, setup_seed, get_logger, visulization_embed
import torch.nn as nn
from models import ConvAttenFusion, MaskGeneEncoderBNTanhCon2Layer, MaskGeneDecoderBNTanhCon2Layer, Aligner
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR, \
    CosineAnnealingLR, SequentialLR, ConstantLR, ExponentialLR
import argparse
from evaluation import eval_model
import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import os, random
from lossfunc import AdaptiveResidual
import json
import pandas as pd
# from lossfunc import REC_loss
from contrastive_loss import contrastive_loss
from sklearn import preprocessing
import subprocess


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def params_setup(dropout):
    params = dict()
    params["in_chans"] = [200, 8]
    params["dims"] = [512, 200]
    params["out_chans"] = [16, 8]
    params["heads"] = [8, 4]
    params["dropout"] = dropout
    low_dim = params["heads"][0] * params["heads"][1]
    # params["inputs"] = [2000, 1024, 512, 128, low_dim, 128, 512, 1024]
    # params["outputs"] = [1024, 512, 128, low_dim, 128, 512, 1024, 2000] 
    params["inputs"] = [2000, 512, 128, low_dim, 128, 1024]
    params["outputs"] = [512, 128, low_dim, 128, 1024, 2000] 

    return params


def logger_initial(logger, params):
    logger.info('params["in_chans"]=[%d, %d]' % (params["in_chans"][0], params["in_chans"][1]))
    logger.info('params["dims"]=[%d, %d]' % (params["dims"][0], params["dims"][1]))
    logger.info('params["out_chans"]=[%d, %d]' % (params["out_chans"][0], params["out_chans"][1]))
    logger.info('params["heads"]=[%d, %d]' %(params["heads"][0], params["heads"][1]))
    logger.info('params["dropout"]=[%.1f, %.1f, %.1f]' %(params["dropout"][0], params["dropout"][1], params["dropout"][2]))
    logger.info('params["inputs"]=%s' % str(params["inputs"]))
    logger.info('params["outputs"]=%s' % str(params["outputs"]))

    return logger


class savePath():
    def __init__(self, proj, time_tab):
        
        partient_dir = "{0}/{1}/".format("TrainProcess", proj)

        self.model_path = partient_dir + time_tab + "/" + "model" + "/"
        check_dir(self.model_path)
        
        self.embed_path = partient_dir + time_tab + "/" + "embed" + "/"
        check_dir(self.embed_path)
        
        self.record_path = partient_dir + time_tab + "/" + "record" + "/"
        check_dir(self.record_path)

        self.log_path = partient_dir + time_tab + "/" + time_tab + ".log"

        self.writer_path = partient_dir + time_tab + "/" + "tensorboard" + "/" + time_tab
        check_dir(self.writer_path)
        self.argument_path = partient_dir + time_tab + "/" + time_tab + ".json"


class TrainingConfig():
    def __init__(self, logger, writer, save_path, args):
        self.logger = logger
        self.writer = writer
        self.model_path = save_path.model_path
        self.embed_path = save_path.embed_path
        self.record_path = save_path.record_path
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.lam = args.lam
        self.scheduler = args.scheduler
        self.savedisk = args.savedisk
        self.smoothing = args.smoothing
        self.tau = args.tau


def train(predict_model, embed_encoder, embed_decoder, aligner, train_iter, test_iter, CLASSES, CE_loss, SIM_loss, REC_loss, optimizer, scheduler, args, last_epoch):
    num_epochs = args.num_epochs
    writer = args.writer
    logger = args.logger
    device = args.device
    
   
    predict_model = predict_model.float().to(device)
    embed_encoder = embed_encoder.float().to(device)
    embed_decoder = embed_decoder.float().to(device)
    aligner = aligner.float().to(device)

    test_last_epoch = np.zeros((last_epoch, 9))
    epoch_count = 0
    round = 1
    stop_count = 0.0
    for epoch in range(num_epochs):
        predict_model.train()
        embed_encoder.train()
        embed_decoder.train()
        loss_, ce_loss_, sim_loss_, rec_loss_, pct_loss_, ect_loss_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        batch_count = 0.0

        for X, UMZ, M, y, MZ, pid, _ in train_iter:

            X = X.float().to(device)

            UMZ = UMZ.float().to(device)
            M = M.float().to(device)
            MZ = MZ.float().to(device)
            y = y.to(device)

            pid_ = preprocessing.LabelEncoder().fit_transform(pid)
            pid__ = torch.as_tensor(pid_).float().to(device)

            GM = torch.sum(torch.abs(MZ), dim=1) > 0
            
            class_ratio = torch.bincount(y) / torch.sum(torch.bincount(y))
            
            CE_loss = nn.CrossEntropyLoss(weight= 1 - class_ratio, reduction="mean", label_smoothing=args.smoothing)

            featfeat_tr, linearprob_tr = predict_model(X)
            embed_gene, pembed_gene = embed_encoder(UMZ)

            rec_gene = embed_decoder(aligner(embed_gene, featfeat_tr))


            ce_loss = CE_loss(linearprob_tr, y)
            mrec = rec_gene * M
            rec_loss = REC_loss(mrec[GM, :], MZ[GM, :])
            
            ect_loss = args.lam * contrastive_loss(pembed_gene[GM, :], pid__[GM], args.tau)
  
            loss = ce_loss + rec_loss + ect_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (scheduler is not None) and (args.scheduler not in  ["MULTILR", "ExponentialLR"]):
                scheduler.step()
                learning_rate = scheduler.get_last_lr()[0]
                writer.add_scalar("LR", learning_rate, round)
                round += 1

            loss_ += loss.item()
            ce_loss_ += ce_loss.item()
            sim_loss_ += 0.0 
            rec_loss_ += rec_loss.item()
            ect_loss_ += ect_loss.item()
            pct_loss_ += 0.0
            
            y_prob = torch.softmax(linearprob_tr, dim=1)

            batch_count += 1
 

            if (epoch + 1) % 5 == 0:
                visulization_embed(embed_gene[GM,:], y[GM], args.embed_path, epoch+1, batch_count, CLASSES)
        
        if args.scheduler == "MULTILR" or args.scheduler == "ExponentialLR":
            scheduler.step()
            learning_rate = scheduler.get_last_lr()[0]
            writer.add_scalar("LR", learning_rate, epoch+1)

        
        """log"""

        eval_train = eval_model(predict_model, train_iter, device)
        eval_test = eval_model(predict_model, test_iter, device)

        if epoch >= num_epochs - last_epoch:
            test_last_epoch[epoch_count, 0] = eval_test[0][1]
            test_last_epoch[epoch_count, 1] = eval_test[0][0]
            test_last_epoch[epoch_count, 2] = eval_test[0][2]
            test_last_epoch[epoch_count, 3] = eval_test[0][4]
            test_last_epoch[epoch_count, 4] = eval_test[0][3]
            test_last_epoch[epoch_count, 5] = eval_test[0][5]
            test_last_epoch[epoch_count, 6] = eval_test[0][7]
            test_last_epoch[epoch_count, 7] = eval_test[0][6]
            test_last_epoch[epoch_count, 8] = eval_test[0][8]
            epoch_count = epoch_count + 1

        if args.savedisk:
            np.save(args.embed_path + "train_embed%d.npy" % (epoch + 1), eval_train[2])
            np.save(args.embed_path + "test_embed%d.npy" % (epoch + 1), eval_test[2])

        """log"""
        logger.info("Epoch[%d/%d], loss:%.4f, CE loss:%.4f, SIM loss:%.4f, REC loss: %.4f, PCON loss: %.4f, ECON loss: %.4f" 
                    % (epoch+1, num_epochs, loss_ / batch_count, ce_loss_ / batch_count, sim_loss_ / batch_count, rec_loss_ / batch_count, 
                       pct_loss_ / batch_count, ect_loss_ / batch_count))
        logger.info("Epoch[%d/%d], train B-acc: %.4f(%.4f-%.4f), test B-acc: %.4f(%.4f-%.4f)"
                    % (epoch+1, num_epochs, eval_train[0][1], eval_train[0][0], eval_train[0][2],
                       eval_test[0][1], eval_test[0][0], eval_test[0][2]))
        logger.info("Epoch[%d/%d], train weighted-f1: %.4f(%.4f-%.4f), test weighted-f1: %.4f(%.4f-%.4f)"
                    % (epoch+1, num_epochs, eval_train[0][4],  eval_train[0][3],  eval_train[0][5], 
                       eval_test[0][4], eval_test[0][3], eval_test[0][5]))
        logger.info("Epoch[%d/%d], train AUC: %.4f(%.4f-%.4f), test AUC: %.4f(%.4f-%.4f)"
                    % (epoch+1, num_epochs, eval_train[0][7], eval_train[0][6], eval_train[0][8], 
                       eval_test[0][7], eval_test[0][6], eval_test[0][8])) 
        
        """tensorboard"""
        writer.add_scalars("LOSS", {"ALL": loss_ / batch_count, "CE": ce_loss_ / batch_count, 
                                    "SIM": sim_loss_ / batch_count, "REC": rec_loss_ / batch_count, 
                                    "PCON": pct_loss_ / batch_count, "ECON": ect_loss_ / batch_count}, epoch + 1)
        writer.add_scalars("ACC", {"B-ACC_train": eval_train[0][1], "B-ACC_test": eval_test[0][1]}, epoch + 1)
        writer.add_scalars("MACRO-F1", {"F1_train": eval_train[0][4], "F1_test": eval_test[0][4]}, epoch + 1)
        writer.add_scalars("AUC", {"AUC_train": eval_train[0][7], "AUC_test": eval_test[0][7]}, epoch + 1)
        

        # if eval_val[4] > best_val_f1:
        if (epoch+1) == num_epochs and args.savedisk:
     
            crr_train_record = eval_train[1]
            crr_test_record = eval_test[1]
            crr_train_erecord = eval_train[3]
            crr_test_erecord = eval_test[3]


            state0 = {
                "model": predict_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": loss_ / batch_count,
                "ce_loss": ce_loss_ / batch_count,
                "sim_loss": sim_loss_ / batch_count,
                "rec_loss": rec_loss_ / batch_count,
                "pct_loss": pct_loss_ / batch_count,
                "ect_loss": ect_loss_ / batch_count,
                "NUM_EPOCHS": num_epochs,
                "DEVICE": device
            }

            torch.save(state0, "{}/predict_model.pt".format(args.model_path))


            state1 = {
                "model": embed_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": loss_ / batch_count,
                "ce_loss": ce_loss_ / batch_count,
                "sim_loss": sim_loss_ / batch_count,
                "rec_loss": rec_loss_ / batch_count,
                "pct_loss": pct_loss_ / batch_count,
                "ect_loss": ect_loss_ / batch_count,
                "NUM_EPOCHS": num_epochs,
                "DEVICE": device
            }

            torch.save(state1, "{}/embed_encoder.pt".format(args.model_path))

            state0 = {
                "model": embed_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": loss_ / batch_count,
                "ce_loss": ce_loss_ / batch_count,
                "sim_loss": sim_loss_ / batch_count,
                "rec_loss": rec_loss_ / batch_count,
                "pct_loss": pct_loss_ / batch_count,
                "ect_loss": ect_loss_ / batch_count,
                "NUM_EPOCHS": num_epochs,
                "DEVICE": device
            }

            torch.save(state1, "{}/embed_decoder.pt".format(args.model_path))

            crr_train_record.to_csv("{}/crr_train_record_epoch_{:d}.csv".format(args.record_path, epoch + 1),
                                    header=True, index=False)
            crr_test_record.to_csv("{}/crr_test_record_epoch_{:d}.csv".format(args.record_path, epoch + 1),
                                header=True, index=False)
            crr_train_erecord.to_csv("{}/crr_train_erecord_epoch_{:d}.csv".format(args.record_path, epoch + 1),
                                    header=True, index=False)
            crr_test_erecord.to_csv("{}/crr_test_erecord_epoch_{:d}.csv".format(args.record_path, epoch + 1),
                                header=True, index=False)   

    return test_last_epoch

def main(args, CLASSES, model, model_architecture, time_tab):
    
    # set path for saving
    save_path = savePath(args.proj, time_tab)
    logger = get_logger(save_path.log_path)
    writer = SummaryWriter(save_path.writer_path)
    
    with open(save_path.argument_path, "w") as fw:
        json.dump(args.__dict__, fw, indent=2)


    PROJECT = args.proj
    batch_size = args.batch_size
    plr = args.plr
    elr = args.elr
    
    traindata = GenomicWSIDataset(PROJECT, "TRAIN", CLASSES, args.mask_ratio)
    testdata = GenomicWSIDataset(PROJECT, "TEST", CLASSES, args.mask_ratio)


    train_iter = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_iter = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    CE_loss = nn.CrossEntropyLoss(reduction="mean",label_smoothing=0.00)
    SIM_loss = AdaptiveResidual(beta=args.beta, C=1)
    REC_loss = nn.MSELoss(reduction="mean")

    predict_model = model[0]
    embed_encoder = model[1]
    embed_decoder = model[2]
    aligner = model[3]

    embed_params = list(embed_encoder.parameters()) + list(embed_decoder.parameters()) + list(aligner.parameters())
    if args.optim == "ADAM":
        optimizer = torch.optim.Adam([{"params": predict_model.parameters()}, 
                                      {"params": embed_params, "lr": elr}], lr=plr, weight_decay=0.01)
    elif args.optim == "RMSprop":
        optimizer = torch.optim.RMSprop([{"params": predict_model.parameters()}, 
                                      {"params": embed_params, "lr": elr}], lr=plr, weight_decay=0.01)
    elif args.optim == "ADAMW":
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        optimizer = torch.optim.AdamW([{"params": predict_model.parameters()}, 
                                      {"params": embed_params, "lr": elr}], lr=plr, weight_decay=0.01)
    elif args.optim == "SGDNesterov":
        optimizer = torch.optim.SGD([{"params": predict_model.parameters()}, 
                                      {"params": embed_params, "lr": elr}], 
                                      lr=plr, nesterov=True, weight_decay=0.01, momentum=0.01)
    else:
        optimizer = torch.optim.SGD([{"params": predict_model.parameters()}, 
                                      {"params": embed_params, "lr": elr}], lr=plr, weight_decay=0.01)
    

    if args.scheduler == "MULTILR":
        scheduler = MultiStepLR(optimizer, milestones=[int(s) for s in args.milestones.split("_")], gamma=args.lrgamma)
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    elif args.scheduler == "OneCycleLR":
        scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=len(train_iter))
    elif args.scheduler == "CyclicLR":
        scheduler = CyclicLR(optimizer, base_lr=elr, max_lr=args.max_lr, mode="exp_range", step_size_up=50, gamma=args.lrgamma)
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
    elif args.scheduler == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, args.lrgamma)
    elif args.scheduler == "SequentialLR_1":
        sch1 = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
        sch2 = ConstantLR(optimizer, factor=0.1, total_iters=args.num_epochs)
        scheduler = SequentialLR(optimizer, [sch1, sch2], milestones=[int(args.milestones.split("_")[0])])
    elif args.scheduler == "SequentialLR_2":    
        sch1 = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=len(train_iter))
        sch2 = ConstantLR(optimizer, factor=0.1, total_iters=args.num_epochs)
        scheduler = SequentialLR(optimizer, [sch1, sch2], milestones=[int(args.milestones.split("_")[0])])
    else:
        scheduler = None 
    

    logger.info("Running PROJECT: %s, P LEARNING RATE: %s, E LEARNING RATE: %s, OPTIMIZER: %s, NUM EPOCHS: %d, BATCH SIZE: %d, DEVICES: %s, SMOOTHING: %s" 
                % (args.proj, str(args.plr), args.elr, args.optim, args.num_epochs, args.batch_size, args.device, str(args.smoothing)))
    

    # record model architecture
    logger = logger_initial(logger, model_architecture)
    train_args = TrainingConfig(logger, writer, save_path, args)
    

    
    test_last_epoch = train(predict_model, embed_encoder, embed_decoder, aligner, train_iter, test_iter, CLASSES, CE_loss, SIM_loss, REC_loss, optimizer, 
                            scheduler, train_args, args.last_epoch)

    return test_last_epoch.mean(axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", type=str, dest="proj", default="CRC-DX", help="POJRECT")
    parser.add_argument("--plr", type=float, dest="plr", default=0.001, help="LEARNING RATE FOR PREDICTED MODEL")
    parser.add_argument("--elr", type=float, dest="elr", default=0.001, help="LEARNING RATE FOR EMBED MODEL")
    parser.add_argument("--num_epochs", type=int, dest="num_epochs", default=200, help="NUM EPOCHS")
    parser.add_argument("--device", type=str, dest="device", default="cuda", help="DEVICE")
    parser.add_argument("--seed", type=int, dest="seed", default=None, help="RANDOM SEED")
    parser.add_argument("--dropout1", type=float, dest="dropout1", default=0.8, help="DROPOUT")
    parser.add_argument("--dropout2", type=float, dest="dropout2", default=0.4, help="DROPOUT")
    parser.add_argument("--dropout3", type=float, dest="dropout3", default=0.4, help="DROPOUT")
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=2048, help="BATCH_SIZE")
    parser.add_argument("--optim", type=str, dest="optim", default="ADAMW", help="OPTIMIZER")
    parser.add_argument("--lam", type=float, dest="lam", default=0.1, help="Trade-OFF BETWEEN LOSS")
    parser.add_argument("--beta", type=float, dest="beta", default=0.5, help="Trade-OFF IN SIM LOSS")
    parser.add_argument("--mask_ratio", type=float, dest="mask_ratio", default=0.7, help="Percentage of masked gene features")
    parser.add_argument("--savedisk", type=bool, dest="savedisk", default=False, help="SAVE INTERMEDIATE OUTPUT")
    
    parser.add_argument("--milestone", type=str, dest="milestones", default="30_50", help="MILESTONES for MULTIPLE LR")
    parser.add_argument("--lrgamma", type=float, dest="lrgamma", default=0.1, help="DECAY RATE OF LR")
    
    parser.add_argument("--T_0", type=int, dest="T_0", default=10, help="T_0 FOR CosineAnnealingWarmRestarts")
    parser.add_argument("--T_mult", type=int, dest="T_mult", default=2, help="T_mult FOR CosineAnnealingWarmRestarts")

    parser.add_argument("--T_max", type=int, dest="T_max", default=10, help="T_max FOR CosineAnnealing")
    parser.add_argument("--max_lr", type=float, dest="max_lr", default=0.1, help="MAX LR FOR CyclicLR & OneCycleLR")
    parser.add_argument("--scheduler", type=str, dest="scheduler", default="None", help="ACTIVATE scheduler")
    parser.add_argument("--last_epoch", type=int, dest="last_epoch", default=10, help="LAST EPOCH FOR EVAL")

    parser.add_argument("--smoothing", type=float, dest="smoothing", default=0.03, help="ALPHA FOR LABEL SMOOTHING")
    parser.add_argument("--tau", type=float, dest="tau", default=0.05, help="TEMPERATURE FOR CONTRASTIVE LOSS")

    parser.add_argument("--EPATH", type=str, dest="path_to_pretrained_embed_model", default=None, help="PATH TO PRETRAINED EMBED MODEL")
    parser.add_argument("--PPATH", type=str, dest="path_to_pretrained_predict_model", default=None, help="PATH TO PRETRAINED PREDICT MODEL")
    
    args = parser.parse_args()

    return args
    

if __name__ == "__main__":

    args = parse_args()
    if args.seed is not None:
        seed_everything(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.proj == "GBM-DX":
        CLASSES = ["Proneural", "Mesenchymal"] 
    else:
        CLASSES = ["MSS", "MSIMUT"]  


    class_num = len(CLASSES)

    time_tab = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("Running time tab:%s" % time_tab)

    # activate tensorboard
    command = "tensorboard --logdir=./TrainProcess/%s/%s" % (args.proj, time_tab)
    process = subprocess.Popen(command)

    # set model architecture
    model_architecture = params_setup([args.dropout1, args.dropout2, args.dropout3])
    
    predict_model = ConvAttenFusion(model_architecture, class_num)
    embed_encoder = MaskGeneEncoderBNTanhCon2Layer(model_architecture)
    embed_decoder = MaskGeneDecoderBNTanhCon2Layer(model_architecture)


    predict_model_zero = ConvAttenFusion(model_architecture, class_num)
    embed_encoder_zero = MaskGeneEncoderBNTanhCon2Layer(model_architecture)
    embed_decoder_zero = MaskGeneDecoderBNTanhCon2Layer(model_architecture)
    aligner = Aligner(model_architecture)


    # load pretrained models
    if args.path_to_pretrained_embed_model is not None:
        load_path = "./TrainProcess/%s/%s/model/embed_encoder.pt" % (args.proj, args.path_to_pretrained_embed_model)
        echeck_point = torch.load(load_path, weights_only=False, map_location = torch.device('cpu'))
        embed_encoder_zero.load_state_dict(echeck_point["model"])
        print("Successfully Load Pretrained EmbedEncoder!!!")

    
    if args.path_to_pretrained_embed_model is not None:
        load_path = "./TrainProcess/%s/%s/model/embed_decoder.pt" % (args.proj, args.path_to_pretrained_embed_model)
        echeck_point = torch.load(load_path, weights_only=False, map_location = torch.device('cpu'))
        embed_decoder_zero.load_state_dict(echeck_point["model"])
        print("Successfully Load Pretrained EmbedDecoder!!!")
    

    if args.path_to_pretrained_predict_model is not None:
        load_path = "./TrainProcess/%s/%s/model/predict_model.pt" % (args.proj, args.path_to_pretrained_predict_model)
        pcheck_point = torch.load(load_path, weights_only=False, map_location = torch.device('cpu'))
        predict_model_zero.load_state_dict(pcheck_point["model"])
        print("Successfully Load Pretrained PredictModel!!!")

    # initialize models with pretrained models
    if args.path_to_pretrained_predict_model is not None:
        print("Init pred_model")
        for (name_a, param_a), (name_b, param_b) in zip(predict_model.named_parameters(), predict_model_zero.named_parameters()):
            if ("ConvAtten0" in name_a and "ConvAtten0" in name_b): #or (name_a == "Classifier.linear.bias"  and  name_b == "Classifier.linear.bias"): 
                if param_a.shape == param_b.shape:
                    param_a.data.copy_(param_b.data)
                    param_a.requires_grad = False
                    print("Initialize and freeze %s" % name_a)

    if args.path_to_pretrained_embed_model is not None:
        print("Init embed_encoder")
        for (name_c, param_c), (name_d, param_d) in zip(embed_encoder.named_parameters(), embed_encoder_zero.named_parameters()):
            if param_c.shape == param_d.shape:
                param_c.data.copy_(param_d.data)
                print("Initialize %s" % name_c)
                # param_c.requires_grad = False
                # print("Initialize and freeze %s" % name_c)

        
    model = [predict_model, embed_encoder, embed_decoder, aligner]

    test_last_epoch_mean = main(args, CLASSES, model, model_architecture, time_tab)
    print(test_last_epoch_mean)

    time.sleep(30)
    process.kill()