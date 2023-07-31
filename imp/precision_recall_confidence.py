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
        
    def interval(self, limit, n):
        
        #probability
        p_hat = self.tp / n
        if p_hat == 1:
            prob = 100 
        else:
            std = ((p_hat * (1-p_hat)) / n) ** 0.5
            z_score = (limit - p_hat)/std
            prob = round(1 - st.norm.cdf(z_score),3)*100
            
        #confidence interval
        alpha = 1-limit
        z_score = st.norm.isf(alpha * 0.5) 
        variance_of_sum = p_hat * (1-p_hat) / n
        std = variance_of_sum ** 0.5
        upper_bound = round(p_hat - z_score * std, 2)
        lower_bound = round(p_hat + z_score * std, 2)
        
        return prob, upper_bound, lower_bound 
        
        
    def recall(self, limit):
        n = self.tp + self.fn
        prob, upper_bound, lower_bound = self.interval(limit, n)
        print(f'With probability {prob}%, recall is greater than {limit}.\n')
        print(f'The {int(limit*100)}% confidence interval for precision is between {upper_bound} and {lower_bound}.')
        
        
    def precision(self, limit):
        n = self.tp + self.fp
        prob, upper_bound, lower_bound = self.interval(limit, n)
        print(f'With probability {prob}%, precision is greater than {limit}.\n')
        print(f'The {int(limit*100)}% confidence interval for precision is between {upper_bound} and {lower_bound}.')
