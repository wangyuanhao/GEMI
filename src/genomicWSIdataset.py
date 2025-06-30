import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.decomposition import PCA
import random
import torch
from sklearn import preprocessing


def resampler(wsi_data, repeat):
    sample_ = [random.choices(list(wsi_data)) for _ in range(repeat)]
    sample = np.squeeze(np.array(sample_))
    return sample

class GenomicWSIDataset(Dataset):
    def __init__(self, PROJECT, MODE, CLASSES, MASK_RATIO):
        

        GENOMIC_PATH = "../genomic/%s/exp_%s_%s.csv" % (PROJECT, "_".join(PROJECT.lower().split("-")), MODE.lower())
        genomic_data = pd.read_csv(GENOMIC_PATH, header=0, index_col=0)

        # get filename of mat (WSI data in mat)
        matfiles = []
        for CLASS in CLASSES:
            WSI_PATH = "../postdata/boostrapping-2t-100-200/mat/%s/%s/%s/" % (PROJECT, MODE, CLASS)
            matfiles_ = [WSI_PATH+file.name for file in os.scandir(WSI_PATH) if file.name.endswith(".mat")]
            matfiles += matfiles_
        
        self.genomic_data = genomic_data
        self.matfiles = matfiles
        self.CLASSES = CLASSES
        self.MASK_RATIO = MASK_RATIO


    def __len__(self):
        return len(self.matfiles)

    def __getitem__(self, item):
        matflie = self.matfiles[item]
        mat = scipy.io.loadmat(matflie)

        wsi_data = mat["bstdfeat"]

        if wsi_data.shape[0] < 200:
            wsi_data = np.vstack((wsi_data, np.zeros((200 - wsi_data.shape[0], wsi_data.shape[1]), dtype="f")))


        for i in range(len(self.CLASSES)):
            if self.CLASSES[i] in matflie:
                label = i

        split_matfile = matflie.split("/")[-1]
        patientID = split_matfile.split("_")[0]

        patient_bool = [True if id==patientID else False for id in self.genomic_data.columns]
        if sum(patient_bool) != 1:
            print(patientID)
            raise "WSI and genomic patientIDs are mismatched!!!"
        genomic_expr = self.genomic_data.iloc[:, patient_bool].values.flatten()

        maskID = np.random.permutation(len(genomic_expr))[0:int(len(genomic_expr)*self.MASK_RATIO)]
        mask = np.zeros_like(genomic_expr)
        mask[maskID] = 1
        unmask = 1 - mask
        mask_genomic_exper = mask * genomic_expr
        unmask_genomic_exper = unmask * genomic_expr

        
        return wsi_data, unmask_genomic_exper, mask, label, mask_genomic_exper, patientID, split_matfile


if __name__ == "__main__":
    
    PROJECT = "GBM-DX"
    MODE = "TRAIN"

    CLASSES = ["Proneural", "Mesenchymal"]
    MASK_RATIO = 0.0

    dataset = GenomicWSIDataset(PROJECT, MODE, CLASSES, MASK_RATIO)
   

    train_iter = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    for X, UMZ, M, y, MZ, pid_, _ in train_iter:
        pid__ = preprocessing.LabelEncoder().fit_transform(pid_)
        pid___ = torch.as_tensor(pid__).float()
        print()