import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import torch 
import torch.nn as nn
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

test_batch_size = 128
def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

def l12_norm(inputs):
    out = torch.mean(torch.sum(torch.sqrt(torch.abs(inputs)), dim=1))
    return out

class MinVolumn(nn.Module):
    def __init__(self, band, num_classes, delta):
        super(MinVolumn, self).__init__()
        self.band = band
        self.delta = delta
        self.num_classes = num_classes
    def __call__(self, edm):
        edm_result = torch.reshape(edm, (self.band,self.num_classes))
        edm_mean = edm_result.mean(dim=1, keepdim=True)
        loss = self.delta * ((edm_result - edm_mean) ** 2).sum() / self.band / self.num_classes
        return loss

class SparseLoss(nn.Module):
    def __init__(self, sparse_decay):
        super(SparseLoss, self).__init__()
        self.sparse_decay = sparse_decay

    def __call__(self, input):
        loss = l12_norm(input)
        return self.sparse_decay*loss

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6,1)

def createConfusionMatrix(y_test,y_pred, plt_name):
    # No of classes for different datasets -> Trento - 6, MUUFL - 11, Houston - 15
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(6),range(6))
    df_cm.columns = ['Buildings','Woods', 'Roads', 'Apples', 'ground', 'Vineyard']
    # df_cm.columns = ["Grass-stressed","Tree","Water","Commercial","Highway","Parking-lot1","Tennis-court","Grass-healthy","Grass-synthetic","Soil","Residential","Road","Railway","Parking-lot2","Running-track"]
    df_cm = df_cm.rename({0:'Buildings',1:'Woods', 2:'Roads', 3:'Apples', 4:'ground', 5:'Vineyard'})
    # df_cm = df_cm.rename({0:"Grass-stressed",1:"Tree",2:"Water",3:"Commercial",4:"Highway",5:"Parking-lot1",6:"Tennis-court",7:"Grass-healthy",8:"Grass-synthetic",9:"Soil",10:"Residential",11:"Road",12:"Railway",13:"Parking-lot2",14:"Running-track"})
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.set(font_scale=0.9)
    plt.figure(figsize=(30,30))
    sns.heatmap(df_cm, cmap="Blues",annot=True,annot_kws={"size": 16}, fmt='g')
    plt.savefig('Cross-HL_'+str(plt_name)+'.eps', format='eps')

def AvgAcc_andEachClassAcc(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    class_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(class_acc)
    return class_acc, average_acc


def result_reports(xtest,xtest2,ytest,name,model, iternum):
    y_pred = np.empty((11700), dtype=np.float32)
    print(xtest.shape,xtest2.shape,ytest.shape)
    number = 11700 // test_batch_size
    print(number)
    for i in range(number):
        temp = xtest[i * test_batch_size:(i + 1) * test_batch_size, :, :]
        temp = temp.cuda()
        temp1 = xtest2[i * test_batch_size:(i + 1) * test_batch_size, :, :]
        temp1 = temp1.cuda()
        temp2,temp4 = model(temp,temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        y_pred[i * test_batch_size:(i + 1) * test_batch_size] = temp3.cpu()
        del temp, temp2, temp3,temp1,temp4

    if (i + 1) * test_batch_size < 11700:
        temp = xtest[(i + 1) * test_batch_size:11700, :, :]
        temp = temp.cuda()
        temp1 = xtest2[(i + 1) * test_batch_size:11700, :, :]
        temp1 = temp1.cuda()
        temp2,temp4 = model(temp,temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        y_pred[(i + 1) * test_batch_size:11700] = temp3.cpu()
        del temp, temp2, temp3,temp1,temp4

    y_pred = torch.from_numpy(y_pred).long()

    overall_acc = accuracy_score(ytest, y_pred)
    confusion_mat = confusion_matrix(ytest, y_pred)
    class_acc, avg_acc = AvgAcc_andEachClassAcc(confusion_mat)
    kappa_score = cohen_kappa_score(ytest, y_pred)
    createConfusionMatrix(ytest, y_pred, str(name)+'_test_'+str(iternum)+'')

    return confusion_mat, overall_acc*100, class_acc*100, avg_acc*100, kappa_score*100
