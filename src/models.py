import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import math
from collections import OrderedDict


class Aligner(nn.Module):
    def __init__(self, params):
        super(Aligner, self).__init__()
        outputs = params["outputs"]
        self.fc1 = nn.Linear(outputs[2], outputs[2])
        self.fc2 = nn.Linear(outputs[2], outputs[2])

    def forward(self, h_G, h_W):
        h = torch.mul(self.fc1(h_G), self.fc2(h_W))
        return h
    
    
    
class ProjectionHead(nn.ModuleDict):
    def __init__(self, inputs, outputs):
        super(ProjectionHead, self).__init__()
        assert len(inputs) == len(outputs)
        layer_num = len(inputs)
        for i in range(layer_num):
            if i == layer_num - 1:
                self.add_module("linear%d" % i, nn.Linear(inputs[i], outputs[i]))
                # self.add_module("BN%d" % i, nn.BatchNorm1d(outputs[i]))
            else:
                self.add_module("linear%d" % i, nn.Linear(inputs[i], outputs[i]))
                # self.add_module("droput%d " % i, nn.Dropout(0.3))
                # self.add_module("BN%d" % i, nn.BatchNorm1d(outputs[i]))
                self.add_module("relu%d" % i, nn.ReLU())

    def forward(self, X):
        x_ = X
        for name, layer in self.items():
            x_ = layer(x_)

        return x_



class MaskGeneEncoderDecoder(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderDecoder, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])
        self.encoder_layer4 = nn.Linear(inputs[3], outputs[3])
        self.decoder_layer1 = nn.Linear(inputs[4], outputs[4])
        self.decoder_layer2 = nn.Linear(inputs[5], outputs[5])
        self.decoder_layer3 = nn.Linear(inputs[6], outputs[6])
        self.decoder_layer4 = nn.Linear(inputs[7], outputs[7])
    
    def forward(self, X):
        layer1 = F.gelu(self.encoder_layer1(X))
        layer2 = F.gelu(self.encoder_layer2(layer1))
        layer3 = F.gelu(self.encoder_layer3(layer2))
        # layer3 = F.relu(layer3_).detach() + F.gelu(layer3_) - F.gelu(layer3_).detach()

        layer4 = self.encoder_layer4(layer3)

        layer5 = F.gelu(self.decoder_layer1(layer4))
        layer6 = F.gelu(self.decoder_layer2(layer5))
        layer7 = F.gelu(self.decoder_layer3(layer6))
        layer8 = self.decoder_layer4(layer7)

        return layer4, layer8


class MaskGeneEncoderDecoderTanh(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderDecoderTanh, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])
        self.encoder_layer4 = nn.Linear(inputs[3], outputs[3])
        self.decoder_layer1 = nn.Linear(inputs[4], outputs[4])
        self.decoder_layer2 = nn.Linear(inputs[5], outputs[5])
        self.decoder_layer3 = nn.Linear(inputs[6], outputs[6])
        self.decoder_layer4 = nn.Linear(inputs[7], outputs[7])
    
    def forward(self, X):
        layer1 = torch.tanh(self.encoder_layer1(X))
        layer2 = torch.tanh(self.encoder_layer2(layer1))
        layer3 = torch.tanh(self.encoder_layer3(layer2))
        # layer3 = F.relu(layer3_).detach() + F.gelu(layer3_) - F.gelu(layer3_).detach()

        layer4 = self.encoder_layer4(layer3)

        layer5 = torch.tanh(self.decoder_layer1(layer4))
        layer6 = torch.tanh(self.decoder_layer2(layer5))
        layer7 = torch.tanh(self.decoder_layer3(layer6))
        layer8 = self.decoder_layer4(layer7)

        return layer4, layer8
    

class MaskGeneEncoderDecoderBNTanh(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderDecoderBNTanh, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.ebn1 = nn.BatchNorm1d(num_features=outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.ebn2 = nn.BatchNorm1d(num_features=outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])
        self.ebn3 = nn.BatchNorm1d(num_features=outputs[2])
        self.encoder_layer4 = nn.Linear(inputs[3], outputs[3])
        self.ebn4 = nn.BatchNorm1d(num_features=outputs[3])

        self.decoder_layer1 = nn.Linear(inputs[4], outputs[4])
        self.dbn1 = nn.BatchNorm1d(num_features=outputs[4])
        self.decoder_layer2 = nn.Linear(inputs[5], outputs[5])
        self.dbn2 = nn.BatchNorm1d(num_features=outputs[5])
        self.decoder_layer3 = nn.Linear(inputs[6], outputs[6])
        self.dbn3 = nn.BatchNorm1d(num_features=outputs[6])
        self.decoder_layer4 = nn.Linear(inputs[7], outputs[7])
     

        # self.projection_head = nn.Sequential(OrderedDict([
        #     ("lin1", nn.Linear(outputs[3], 64)),
        #     ("gelu1", nn.Tanh()),
        #     ("lin2", nn.Linear(64, 128))
        # ]))
    
    def forward(self, X):
        layer1 = torch.tanh(self.ebn1(self.encoder_layer1(X)))
        layer2 = torch.tanh(self.ebn2(self.encoder_layer2(layer1)))
        layer3 = torch.tanh(self.ebn3(self.encoder_layer3(layer2)))
        # layer3 = F.relu(layer3_).detach() + F.gelu(layer3_) - F.gelu(layer3_).detach()

        layer4 = self.encoder_layer4(layer3)

        layer5 = torch.tanh(self.dbn1(self.decoder_layer1(layer4)))
        layer6 = torch.tanh(self.dbn2(self.decoder_layer2(layer5)))
        layer7 = torch.tanh(self.dbn3(self.decoder_layer3(layer6)))
        layer8 = self.decoder_layer4(layer7)

        # player4 = self.projection_head(layer4)

        return layer4, layer8


class MaskGeneEncoderDecoder2Layer(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderDecoder2Layer, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])
        self.encoder_layer4 = nn.Linear(inputs[3], outputs[3])
        self.decoder_layer1 = nn.Linear(inputs[4], outputs[4])
        self.decoder_layer2 = nn.Linear(inputs[5], outputs[5])
        self.decoder_layer3 = nn.Linear(inputs[6], outputs[6])
        self.decoder_layer4 = nn.Linear(inputs[7], outputs[7])
    
    def forward(self, X):
        layer1 = F.gelu(self.encoder_layer1(X))
        layer2 = F.gelu(self.encoder_layer2(layer1))
        layer3 = F.gelu(self.encoder_layer3(layer2))
        # layer3 = F.relu(layer3_).detach() + F.gelu(layer3_) - F.gelu(layer3_).detach()

        layer4 = self.encoder_layer4(layer3)

        layer5 = F.gelu(self.decoder_layer1(layer4))
        layer6 = F.gelu(self.decoder_layer2(layer5))
        layer7 = F.gelu(self.decoder_layer3(layer6))
        layer8 = self.decoder_layer4(layer7)

        return layer4, layer8


class MaskGeneEncoderDecoderTanh2Layer(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderDecoderTanh2Layer, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])

        self.decoder_layer1 = nn.Linear(inputs[4], outputs[4])
        self.decoder_layer2 = nn.Linear(inputs[5], outputs[5])
        self.decoder_layer3 = nn.Linear(inputs[6], outputs[6])

    
    def forward(self, X):
        layer1 = torch.tanh(self.encoder_layer1(X))
        layer2 = torch.tanh(self.encoder_layer2(layer1))
        layer3 = self.encoder_layer3(layer2)
        # layer3 = F.relu(layer3_).detach() + F.gelu(layer3_) - F.gelu(layer3_).detach()

        layer4 = torch.tanh(self.decoder_layer1(layer3))
        layer5 = torch.tanh(self.decoder_layer2(layer4))
        layer6 = self.decoder_layer3(layer5)

        return layer3, layer6
    

class MaskGeneEncoderDecoderBNTanh2Layer(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderDecoderBNTanh, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.ebn1 = nn.BatchNorm1d(num_features=outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.ebn2 = nn.BatchNorm1d(num_features=outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])
        self.ebn3 = nn.BatchNorm1d(num_features=outputs[2])


        self.decoder_layer1 = nn.Linear(inputs[4], outputs[4])
        self.dbn1 = nn.BatchNorm1d(num_features=outputs[4])
        self.decoder_layer2 = nn.Linear(inputs[5], outputs[5])
        self.dbn2 = nn.BatchNorm1d(num_features=outputs[5])
        self.decoder_layer3 = nn.Linear(inputs[6], outputs[6])

     

        # self.projection_head = nn.Sequential(OrderedDict([
        #     ("lin1", nn.Linear(outputs[3], 64)),
        #     ("gelu1", nn.Tanh()),
        #     ("lin2", nn.Linear(64, 128))
        # ]))
    
    def forward(self, X):
        layer1 = torch.tanh(self.ebn1(self.encoder_layer1(X)))
        layer2 = torch.tanh(self.ebn2(self.encoder_layer2(layer1)))
        layer3 = self.ebn3(self.encoder_layer3(layer2))
        # layer3 = F.relu(layer3_).detach() + F.gelu(layer3_) - F.gelu(layer3_).detach()

  

        layer4 = torch.tanh(self.dbn1(self.decoder_layer1(layer3)))
        layer5 = torch.tanh(self.dbn2(self.decoder_layer2(layer4)))
        layer6 = self.decoder_layer4(layer5)

        # player4 = self.projection_head(layer4)

        return layer3, layer6
    

class MaskGeneEncoderDecoderBNTanhCon2Layer(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderDecoderBNTanhCon2Layer, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.ebn1 = nn.BatchNorm1d(num_features=outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.ebn2 = nn.BatchNorm1d(num_features=outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])



        self.decoder_layer1 = nn.Linear(inputs[3], outputs[3])
        self.dbn1 = nn.BatchNorm1d(num_features=outputs[3])
        self.decoder_layer2 = nn.Linear(inputs[4], outputs[4])
        self.dbn2 = nn.BatchNorm1d(num_features=outputs[4])
        self.decoder_layer3 = nn.Linear(inputs[5], outputs[5])

     
        self.projection_head = nn.Sequential(OrderedDict([
            ("lin1", nn.Linear(outputs[2], 64)),
            ("bn1", nn.BatchNorm1d(64)),
            ("relu", nn.Tanh()),
            ("lin2", nn.Linear(64, 128)),
            ("bn2", nn.BatchNorm1d(128))
        ]))
    
    def forward(self, X):
        layer1 = torch.tanh(self.ebn1(self.encoder_layer1(X)))
        layer2 = torch.tanh(self.ebn2(self.encoder_layer2(layer1)))
       
        layer3 = self.encoder_layer3(layer2)


        layer4 = torch.tanh(self.dbn1(self.decoder_layer1(layer3)))
        layer5 = torch.tanh(self.dbn2(self.decoder_layer2(layer4)))
        layer6 = self.decoder_layer3(layer5)

        player3 = self.projection_head(layer3)

        return layer3, player3, layer6
    
    

class MaskGeneEncoderBNTanhCon2Layer(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderBNTanhCon2Layer, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.ebn1 = nn.BatchNorm1d(num_features=outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.ebn2 = nn.BatchNorm1d(num_features=outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])

        self.projection_head = nn.Sequential(OrderedDict([
            ("lin1", nn.Linear(outputs[2], 64)),
            ("bn1", nn.BatchNorm1d(64)),
            ("relu", nn.Tanh()),
            ("lin2", nn.Linear(64, 128)),
            ("bn2", nn.BatchNorm1d(128))
        ]))
    
    def forward(self, X):
        layer1 = torch.tanh(self.ebn1(self.encoder_layer1(X)))
        layer2 = torch.tanh(self.ebn2(self.encoder_layer2(layer1)))
       
        layer3 = self.encoder_layer3(layer2)


        player3 = self.projection_head(layer3)

        return layer3, player3
    
class MaskGeneDecoderBNTanhCon2Layer(nn.Module):
    def __init__(self, params):
        super(MaskGeneDecoderBNTanhCon2Layer, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)


        self.decoder_layer1 = nn.Linear(inputs[3], outputs[3])
        self.dbn1 = nn.BatchNorm1d(num_features=outputs[3])
        self.decoder_layer2 = nn.Linear(inputs[4], outputs[4])
        self.dbn2 = nn.BatchNorm1d(num_features=outputs[4])
        self.decoder_layer3 = nn.Linear(inputs[5], outputs[5])

     
    def forward(self, hidden):
        layer4 = torch.tanh(self.dbn1(self.decoder_layer1(hidden)))
        layer5 = torch.tanh(self.dbn2(self.decoder_layer2(layer4)))
        layer6 = self.decoder_layer3(layer5)


        return layer6
       
class MaskGeneEncoderDecoderBNTanhCon(nn.Module):
    def __init__(self, params):
        super(MaskGeneEncoderDecoderBNTanhCon, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.ebn1 = nn.BatchNorm1d(num_features=outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.ebn2 = nn.BatchNorm1d(num_features=outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])
        self.ebn3 = nn.BatchNorm1d(num_features=outputs[2])
        self.encoder_layer4 = nn.Linear(inputs[3], outputs[3])
        # self.ebn4 = nn.BatchNorm1d(num_features=outputs[3])

        self.decoder_layer1 = nn.Linear(inputs[4], outputs[4])
        self.dbn1 = nn.BatchNorm1d(num_features=outputs[4])
        self.decoder_layer2 = nn.Linear(inputs[5], outputs[5])
        self.dbn2 = nn.BatchNorm1d(num_features=outputs[5])
        self.decoder_layer3 = nn.Linear(inputs[6], outputs[6])
        self.dbn3 = nn.BatchNorm1d(num_features=outputs[6])
        self.decoder_layer4 = nn.Linear(inputs[7], outputs[7])
     

        self.projection_head = nn.Sequential(OrderedDict([
            ("lin1", nn.Linear(outputs[3], 64)),
            #("bn1", nn.BatchNorm1d(64)),
            ("relu", nn.Tanh()),
            ("lin2", nn.Linear(64, 128))
            #("bn2", nn.BatchNorm1d(128))
        ]))
    
    def forward(self, X):
        # layer1 = torch.tanh(self.ebn1(self.encoder_layer1(X)))
        layer1 = torch.tanh(self.ebn1(self.encoder_layer1(X)))
        # layer2 = torch.tanh(self.ebn2(self.encoder_layer2(layer1)))
        layer2 =  torch.tanh(self.ebn2(self.encoder_layer2(layer1)))
        # layer3 = torch.tanh(self.ebn3(self.encoder_layer3(layer2)))
        layer3 =  torch.tanh(self.ebn3(self.encoder_layer3(layer2)))
        # layer3 = F.relu(layer3_).detach() + F.gelu(layer3_) - F.gelu(layer3_).detach()

        layer4 = self.encoder_layer4(layer3)

        # layer5 = torch.tanh(self.dbn1(self.decoder_layer1(layer4)))
        layer5 =  torch.tanh(self.dbn1(self.decoder_layer1(layer4)))
        # layer6 = torch.tanh(self.dbn2(self.decoder_layer2(layer5)))
        layer6 =  torch.tanh(self.dbn2(self.decoder_layer2(layer5)))
        # layer7 = torch.tanh(self.dbn3(self.decoder_layer3(layer6)))
        layer7 =  torch.tanh(self.dbn3(self.decoder_layer3(layer6)))
        layer8 = self.decoder_layer4(layer7)

        player4 = self.projection_head(layer4)

        return layer4, player4, layer8
    
    
class OneLayerClassifier(nn.Module):
    def __init__(self, D, class_num):
        super(OneLayerClassifier, self).__init__()
        self.linear = nn.Linear(D, class_num)

    def forward(self, x):
        x_ = self.linear(x)
        return x_


class TwoLayerClassifier(nn.Module):
    def __init__(self, D, class_num, dropout):
        super(TwoLayerClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D,  int(D/2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(D/2), class_num)
        )

    def forward(self, x):
        x_ = self.encoder(x)
        return x_


class FittingBlocks(nn.Module):
    def __init__(self, inputs, outputs, dropout):
        super(FittingBlocks, self).__init__()
        self.FitBlocks = nn.Sequential(
            nn.Linear(inputs, outputs),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        z = self.FitBlocks(x)
        return z


class PatchScoreNet(nn.ModuleDict):
    def __init__(self, fc_inputs, fc_outputs, ydim, dropout):
        super(PatchScoreNet, self).__init__()
        for i in range(len(fc_inputs)):
            if i == (len(fc_inputs) - 1):
                dropout = 0.0
            self.add_module("fitblock%d"% (i+1), FittingBlocks(fc_inputs[i], fc_outputs[i], dropout))

        self.add_module("regressor", nn.Linear(fc_outputs[i], ydim))

    def forward(self, x):
        z = None
        x_ = x
        for name, layer in self.items():
            if name == "regressor":
                z = layer(x_)
            else:
                x_ = layer(x_)

        return x_, z


class SimpleAttention(nn.Module):
    def __init__(self, D, H):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(D, H),
            nn.Tanh()
        )

    def forward(self, X):
        atten = self.attention(X)
        return atten



class DotProductAttention(nn.Module):
    def __init__(self, D, H):
        super(DotProductAttention, self).__init__()
        self.D = D 
        self.H = H 
        scaling_weight = []

        for i in range(H):
            weight = nn.Parameter(torch.Tensor(D))
            torch.nn.init.normal_(weight)
            scaling_weight.append(weight)  

        in_place_scaling_weight = nn.Parameter(torch.Tensor(D))
        # torch.nn.init.uniform_(in_place_scaling_weight, -1, 1)
        torch.nn.init.constant_(in_place_scaling_weight, 1/D)
        self.in_place_scaling_weight = in_place_scaling_weight
        self.scaling_weight = scaling_weight

    def forward(self, X):
        atten_weight = []
        for i in range(self.H):
            scaling_X_ = torch.mul(X, self.scaling_weight[i].to(X.device))
            weight_ = torch.bmm(scaling_X_, torch.transpose(scaling_X_, 1, 2)) / torch.tensor(X.shape[2])
            weight = weight_ - torch.diag_embed(torch.diagonal(weight_, dim1=1, dim2=2))
            average_weight = torch.sum(weight, dim=2, keepdim=True)
            atten_weight.append(average_weight)
        
        atten_weight_ = torch.cat(atten_weight, dim=2)
        X_ = torch.mul(X, self.in_place_scaling_weight.to(X.device))

        return atten_weight_, X_
        
        

class ConvBlock(nn.Module):
    def __init__(self, in_chan, D, out_chan, dropout):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(D, out_chan, bias=False),
            nn.BatchNorm1d(in_chan),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_ = self.conv(x)

        return x_


class ConvAttenBlock(nn.Module):
    def __init__(self, in_chan, D, out_chan, heads, dropout):
        super(ConvAttenBlock, self).__init__()
        self.conv = ConvBlock(in_chan, D, out_chan, dropout)
        # self.attention = SimpleAttention(in_chan, heads)
        self.attention = DotProductAttention(in_chan, heads)
        # self.dropout = nn.Dropout(dropout)
        # self.layernorm = nn.LayerNorm(in_chan)

    def forward(self, X):
        z = self.conv(X)
        z_ = z.transpose(1, 2)
        atten, z__ = self.attention(z_)
        atten_ = torch.softmax(atten, dim=1)
        # z___ = torch.bmm(atten_.transpose(1, 2), self.dropout(z__))
        z___ = torch.bmm(atten_.transpose(1, 2), z__)
        # z___ = self.layernorm(z___)

        # z_ = z.transpose(1, 2)
        # atten = self.attention(z_)
        # atten_ = torch.softmax(atten, dim=1)
        # z__ = torch.bmm(atten_.transpose(1, 2), z_)

        return z___


class ConvAttenFusion(nn.ModuleDict):
    """[100, 512, 16, 8]"""
    """[8, 100, 8, 4]"""
    # def __init__(self, in_chan, D, out_chan, heads, dropout, pinputs, poutputs):
    def __init__(self, params, class_num):
        in_chan = params["in_chans"]
        D = params["dims"]
        out_chan = params["out_chans"]
        heads = params["heads"]
        dropout = params["dropout"]
        # pinputs = params["pinputs"]
        # poutputs = params["poutputs"]
        super(ConvAttenFusion, self).__init__()
        layers = len(in_chan)
        for i in range(layers):
            self.add_module("ConvAtten%d" % i, ConvAttenBlock(in_chan[i], D[i],
                                                              out_chan[i], heads[i], dropout[i]))

        flatten_dim = heads[-1] * heads[-2]
        self.add_module("Classifier", OneLayerClassifier(flatten_dim, class_num))
        # self.add_module("Classifier", TwoLayerClassifier(flatten_dim, class_num, dropout[-1]))
        # self.add_module("Projection", ProjectionHead(pinputs, poutputs))

    def forward(self, X):
        z = None
        # projection = None
        x__ = None
        x_ = deepcopy(X)
        for name, layer in self.items():
            if name == "Classifier":
                # x_  = F.relu(x_).detach() + F.gelu(x_) - F.gelu(x_).detach()
                x__ = x_.view(x_.shape[0], -1)
                z = layer(x__)
            # elif name == "Projection":
            #     px__ = x_.view(x_.shape[0], -1)
            #     px__M = torch.cat((px__, M), dim=1)
            #     projection = layer(px__M)
            else:
                x_ = layer(x_)

        # return projection, px__, z
        return x__, z


# class ConvFusion(nn.Module):
#     def __init__(self, WD, d, CD1, CD2, dropout=0.0):
#         super(ConvFusion, self).__init__()
#         self.conv1d = nn.Sequential(
#             nn.Linear(WD, d, bias=False),
#             nn.BatchNorm1d(CD1),
#             nn.Tanh(),
#             nn.Dropout(dropout)
#         )
#         # self.conv1d = nn.Linear(WD, 1)
#         self.attention = SimpleAttention(CD1, 1)
#         # self.classifier = TwoLayerClassifier(CD1, CD2, dropout)
#         self.classifier = OneLayerClassifier(CD1)
#
#     def forward(self, X):
#         z = self.conv1d(X)
#         z_ = z.transpose(1, 2)
#         atten = self.attention(z_)
#         atten_ = torch.softmax(atten, dim=1)
#         z__ = torch.bmm(atten_.transpose(1, 2), z_).squeeze()
#         score = self.classifier(z__)
#         return score

class MILAttenBlock(nn.Module):
    def __init__(self, D, out_chan, head, dropout):
        super(MILAttenBlock, self).__init__()
        self.D, self.L = D, out_chan
        self.attention = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(self.L, head)
        )

    def forward(self, X):
        A = self.attention(X)
        W = F.softmax(A, dim=1)
        z = torch.bmm(X.transpose(1, 2), W)
        z_ = z.squeeze()
        return z_


class GatedAttenBlock(nn.Module):
    def __init__(self, D, L, dropout):
        super(GatedAttenBlock, self).__init__()
        self.D, self.L = D, L

        self.attention_U = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.weight = nn.Linear(self.L, 1)


    def forward(self, X):
        A_U = self.attention_U(X)
        A_V = self.attention_V(X)
        W_ = self.weight(A_V * A_U)
        W = F.softmax(W_, dim=1)
        z = torch.bmm(X.transpose(1, 2), W)
        z_ = z.squeeze()
        return z_


class MILAttenFusion(nn.ModuleDict):

    def __init__(self, params, class_num):
        ##
        # out_chan = [16, 8]
        # D = [512, 512]
        # heads = [8, 4]
        # dropout = [0.8, 0.4]
        ##

        super(MILAttenFusion, self).__init__()
        out_chan = params["out_chans"]
        D = params["dims"]
        dropout = params["dropout"]
        heads = params["heads"]

        self.mil_atten_layer1 = MILAttenBlock(D[0], out_chan[0], heads[0], dropout[0])
        self.mil_atten_layer2 = MILAttenBlock(D[1], out_chan[1], heads[1], dropout[1])

        self.flatten_dim = heads[-1]*D[-1]
        self.linear = nn.Linear(self.flatten_dim, 32)
        self.classifier = nn.Linear(32, class_num)


    def forward(self, X):
        z = None
        x_ = deepcopy(X)

        mil_atten_layer1_x_ = self.mil_atten_layer1(x_)
        mil_atten_layer1_x = mil_atten_layer1_x_.transpose(1, 2)

        mil_atten_layer2_x_ = self.mil_atten_layer2(mil_atten_layer1_x)
        mil_atten_layer2_x = mil_atten_layer2_x_.view(mil_atten_layer2_x_.shape[0], -1)
        x__ = self.linear(mil_atten_layer2_x)

        z = self.classifier(x__)

        return x__, z


class GMILAttenFusion(nn.ModuleDict):

    def __init__(self, D, L, CD,pinputs, poutputs, dropout):
        super(GMILAttenFusion, self).__init__()
        self.add_module("GAttention", GatedAttenBlock(D, L, dropout[0]))
        # self.add_module("Classifier", OneLayerClassifier(D))
        self.add_module("Classifier", TwoLayerClassifier(D, CD, dropout[1]))
        self.add_module("Projection", ProjectionHead(pinputs, poutputs))

    def forward(self, X):
        z = None
        projection = None
        x_ = deepcopy(X)
        for name, layer in self.items():
            if name == "Classifier":
                z = layer(x_)
            elif name == "Projection":
                projection = layer(x_)
            else:
                x_ = layer(x_)

        return projection, x_, z


class ConvFusion(nn.ModuleDict):
    def __init__(self, params, class_num):
        ## 
        # in_chan = [200, 16]
        # out_chan = [16, 8]
        # D = [512, 200]
        # dropout = [0.8, 0.4]
        ## 
        super(ConvFusion, self).__init__()
        in_chan = params["in_chans"]
        out_chan = params["out_chans"]
        D = params["dims"]
        dropout = params["dropout"]
        self.conv1d_layer1 = ConvBlock(in_chan[0], D[0], out_chan[0], dropout[0])
        self.conv1d_layer2 = ConvBlock(in_chan[1], D[1], out_chan[1], dropout[1])
        flatten_dim = out_chan[0] * out_chan[1]
        self.classifier = OneLayerClassifier(flatten_dim, class_num)
        self.projector = nn.Linear(flatten_dim, 32)

    def forward(self, X):
        z = None
        x__ = None
        x_ = deepcopy(X)
        conv1d_layer1_x_ = self.conv1d_layer1(x_)
        conv1d_layer1_x = conv1d_layer1_x_.transpose(1, 2)
        conv1d_layer2_x_ = self.conv1d_layer2(conv1d_layer1_x)
        x__ = conv1d_layer2_x_.view(conv1d_layer2_x_.shape[0], -1)
        x___ = self.projector(x__)
        z = self.classifier(x__)

        return x___, z






