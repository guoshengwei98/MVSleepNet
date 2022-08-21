# A package from the Internet 

import numpy as np
import matplotlib.pyplot as plt


class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def getMatrix(self,normalize=True):
        if normalize:
            per_sum = self.matrix.sum(axis=1)  
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   
            self.matrix=np.around(self.matrix, 2)  
            self.matrix[np.isnan(self.matrix)] = 0  
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  
        plt.title("Normalized confusion matrix")  
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name) 
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45) 

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  

        plt.tight_layout()  

        plt.colorbar() 
        plt.savefig('./ConfusionMatrix.png', bbox_inches='tight')  
        plt.show()