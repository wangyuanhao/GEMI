import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from bootstrap import bootstrap_ap, bootstrap, bootstrap_all


def get_sen_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity

def ensemble_patient(patientID, y_pred, y_true):
    # patientID: a list with patient id
    # y_pred: prediction score of a sample
    # y_true: ground-truth label of a sample
    uq_patientID = list(set(patientID))

    ensemble_y_pred = []
    ensemble_y_true = []

    for uq_id in uq_patientID:
        bool_idx = [True if uq_id == id else False for id in patientID]
        ensemble_y_pred.append(y_pred[bool_idx].mean())
        ensemble_y_true.append(y_true[bool_idx].mean())

    ensemble_y_true = np.array(ensemble_y_true).ravel()
    ensemble_y_pred = np.array(ensemble_y_pred).ravel()

    return uq_patientID, ensemble_y_pred, ensemble_y_true

def full_evaluation(y_true, y_pred):
    # acc = np.sum((y_pred >= 0.5)==y_true) / len(y_true)
    # auc_ = bootstrap(y_true, y_pred, 500, 0.95)
    # # ap_ = bootstrap_ap(y_true, y_pred, 500, 0.95)
    #
    # f1_score_ = f1_score(y_true, y_pred >= 0.5)
    # sens, spec = get_sen_spec(y_true, y_pred >= 0.5)
    res = bootstrap_all(y_true, y_pred, 500, 0.95)

    return res


#---------------------MetaCon------------------------------#
# record_fi = "./CRC_DX.csv"
# proj = "CRC DX"
# records = pd.read_csv(record_fi, header=0, index_col=None)
# y_true = records["target"].values
# y_pred = records["score"].values
#----------------------------------------------------------#

#--------------------Ours----------------------------------#
record_fi = "./STAD_DX_1DConv_2022.csv"
proj = "STAD DX"
records = pd.read_csv(record_fi, header=0, index_col=None)
patient_id = records["patientID"].tolist()
y_true = records["y_true"].values
y_pred = records["y_pred"].values
_, ey_pred, ey_true = ensemble_patient(patient_id, y_pred, y_true)
#----------------------------------------------------------#


#---------------------ResNet18-----------------------------#
# record_fi = "./CRC_DX_R18.csv"
# id_fi = "./CRC_DX_test.csv"
# proj = "CRC DX"
# records = pd.read_csv(record_fi, header=0, index_col=None)
# labels = pd.read_csv(id_fi, header=0, index_col=None)
#
# y_true = []
# y_pred = []
#
# record_sampleID = records["PatientID"].tolist()
# label_sampleID = labels["patientID"].tolist()
#
# for i in range(len(record_sampleID)):
#     if record_sampleID[i] in label_sampleID:
#         label_idx = label_sampleID.index(record_sampleID[i])
#         y_true_ = labels.iloc[label_idx, 1]
#         y_true.append(y_true_)
#         y_pred_ = records.iloc[i, 2]
#         y_pred.append(y_pred_)
#
# y_true = np.array(y_true)
# y_pred = np.array(y_pred)
#--------------------------------------------------------------#

auc_l, auc_m, auc_h, acc_l, acc_m, acc_h, f1_l, f1_m, f1_h = full_evaluation(ey_true, ey_pred)

print("In %s, test AUC: %.4f(%.4f-%.4f), test ACC: %.4f(%.4f-%.4f), test F1: %.4f(%.4f-%.4f)"
      % (proj, auc_m, auc_l, auc_h, acc_m, acc_l, acc_h, f1_m, f1_l, f1_h))
