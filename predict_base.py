import torch
import numpy as np
from dataset import load_dataset
from lw import LightningWrapper_base
import torch.nn.functional as F
from sklearn import metrics
from matrix import DrawConfusionMatrix

labels_name=['W', 'N1', 'N2', 'N3', 'REM']
drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name) 

mylw = LightningWrapper_base()
model = mylw.load_from_checkpoint('model-base.ckpt')
model.eval()

val_dataset = load_dataset()
val_preds,val_labels=[],[]
for i in range(len(val_dataset)):
    val_data,val_label = val_dataset[i]
    val_data=val_data.unsqueeze(0)

    val_pred = model(val_data)
    val_pred=val_pred.squeeze(0)
    val_pred = F.softmax(val_pred, dim=1)
    val_pred = torch.argmax(val_pred, dim=1)
    val_pred=val_pred.unsqueeze(1)
    val_labels.append(val_label)
    val_preds.append(val_pred)

label = np.concatenate(val_labels)
pred = np.concatenate(val_preds)

acc = (pred == label).sum() / len(pred)
kappa = metrics.cohen_kappa_score(label, pred)
f1_score = metrics.f1_score(label, pred, average=None)
Mf1 = f1_score.sum() /len(f1_score)
print("total:",len(pred),"\nacc:",acc,"\nkappa:",kappa,"\nF1_score:",f1_score,"\nMF1:",Mf1)

drawconfusionmatrix.update(pred, label)  
drawconfusionmatrix.drawMatrix() 
confusion=drawconfusionmatrix.getMatrix() 
print("confusion_matrix:\n",confusion)