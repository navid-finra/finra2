#----------------------------------------------------------------------------------------------------------------#
from error_analysis import err_cluster
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import chi2_contingency
from scipy.stats import kstest
import numpy as np
#----------------------------------------------------------------------------------------------------------------#

class continuous_variable(err_cluster):
    def __init__(self, df, train_split_index, labels_column, model):
        super().__init__(df, train_split_index, labels_column, model)

    def r2(self):
        self.Error_analysis_train()    
        r2 = r2_score(self.y_train, self.svm_pred)
        print('r2 score for perfect model is', r2)

    def mse(self):
        self.Error_analysis_train()    
        r2 = mean_squared_error(self.y_train, self.svm_pred)
        print('r2 score for perfect model is', r2)

    def chi_square(self):
        self.Error_analysis_train()     
        data = [self.y_train.tolist(), self.svm_pred.tilist()]
        chi_square_dict = {'stats':chi2_contingency(data)[0], 'p':chi2_contingency(data)[1],'dof':chi2_contingency(data)[2],
        'expexted':chi2_contingency(data)[3]}
        print(chi_square_dict)
    
    def Kolmogorov_Smirnov(self):
        self.Error_analysis_train()
        print(kstest(np.array(self.y_train), 'norm'))

#----------------------------------------------------------------------------------------------------------------#
