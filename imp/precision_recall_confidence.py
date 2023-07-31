import scipy.stats as st
from sklearn import metrics
import pandas as pd

class Precision_Recall_Confidence:
    def __init__(self, data):
        self.data = data
        self.predict = self.data.model.predict(self.data.x_test)
        conf_mat = metrics.confusion_matrix(self.data.y_test, self.predict)
        self.tn = conf_mat[0][0]
        self.fp = conf_mat[0][1]
        self.fn = conf_mat[1][0]
        self.tp = conf_mat[1][1]
        
        
    def recall(self, limit):
        n = self.tp + self.fn
        p_hat = self.tp / n
        
        if p_hat == 1:
            prob = 100 
        else:
            std = ((p_hat * (1-p_hat)) / n) ** 0.5
            z_score = (limit - p_hat)/std
            prob = round(1 - st.norm.cdf(z_score),3)*100
        print(f'With probability {prob}%, recall is greater than {limit}')
        
        
    def precision(self, limit):
        n = self.tp + self.fp
        p_hat = self.tp / n
        
        if p_hat == 1:
            prob = 100 
        else:
            std = ((p_hat * (1-p_hat)) / n) ** 0.5
            z_score = (limit - p_hat)/std
            prob = round(1 - st.norm.cdf(z_score),3)*100
        print(f'With probability {prob}%, precision is greater than {limit}')
