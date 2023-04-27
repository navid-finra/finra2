#----------------------------------------------------------------------------------------------------------------#
from Error_analysis import Err_Cluster
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import chi2_contingency
from scipy.stats import kstest
import numpy as np
#----------------------------------------------------------------------------------------------------------------#

class Continuous_Variable(Err_Cluster):
    def __init__(self,x_train,x_test,y_train,y_test,model):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

        self.Error_analysis(self.x_train, self.y_train)
        model_pred = self.model_pred
        self.model_pred = model_pred

#----------------------------------------------------------------------------------------------------------------#

    def r2(self):
        self.Error_analysis_train()    
        r2 = r2_score(self.y_train, self.model_pred)
        print('r2 score for perfect model is', r2)

    def mse(self):
        self.Error_analysis_train()    
        mse = mean_squared_error(self.y_train, self.model_pred)
        print('mse score for perfect model is', mse)

    def chi_square(self):
        self.Error_analysis_train()     
        data = [self.y_train.tolist(), self.model_pred.tolist()]
        chi_square_dict = {'stats':chi2_contingency(data)[0], 'p':chi2_contingency(data)[1],'dof':chi2_contingency(data)[2],
        'expexted':chi2_contingency(data)[3]}
        print(chi_square_dict)
    
    def Kolmogorov_Smirnov(self):
        print(kstest(np.array(self.y_train), 'norm'))

#----------------------------------------------------------------------------------------------------------------#
