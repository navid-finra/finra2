import pandas as pd
from aif360.sklearn.metrics import disparate_impact_ratio
from sklearn import metrics

class post_training:
    def __init__(self, data):
        
        self.data = data
        self.predict = self.data.model.predict(self.data.x_train)
        conf_mat = metrics.confusion_matrix(self.data.y_train, self.predict)
        self.tn = conf_mat[0][0]
        self.fp = conf_mat[0][1]
        self.fn = conf_mat[1][0]
        self.tp = conf_mat[1][1]

    def Disparate_Impact_Ratio(self):
        print('Disparate Impact Ratio:', round(disparate_impact_ratio(pd.Series(self.data.y_train), pd.Series(self.predict)),2))
        
    def Conditional_Acceptance(self):
        print('DCAcc:', round(sum(self.data.y_train)/sum(self.predict),2))
        
    def Conditional_Rejection(self):
        print('DCR:', round((len(self.data.y_train)-sum(self.data.y_train))/(len(self.predict)-sum(self.predict)),2))
        
    def Specificity(self):
        print('Specificity difference:', round(self.tn/(self.tn + self.fp),2))
        
    def Acceptance_Rates(self):
        print('DAR:', round(self.tp/(self.tp + self.fp),2))
        
    def Rejection_Rates(self):
        print('DRR:', round(self.tn/(self.tn + self.fn),2))
        
    def Treatment_Equality(self):
        print('TE:', round(self.fn/self.fp,2))
        
    def Accuracy(self):
        print('Accuracy:', round((self.tp+self.tn)/(self.tp+self.fn+self.tn+self.fp)))
        
    def Recall(self):
        print('Accuracy:', round(self.tp/(self.tp+self.fn)))