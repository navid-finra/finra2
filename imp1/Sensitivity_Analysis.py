#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
#----------------------------------------------------------------------------------------------------------------#

class Sensitivity_Analysis_class:
    def __init__(self,x_train,x_test,y_train,y_test,model):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

  
    def Sensitivity_Analysis(self):
        interval= 40 #assumed
        numbers = [interval*float(x)/10 for x in range(-10 , 11)]

        if len(numbers)%2 == 0:
            counter1 = len(numbers)/2
            counter2 = counter1+1
        else:
            counter1 = (len(numbers)+1)/2
            counter2 = (len(numbers)-3)/2
        full=[]
        for i in range(len(self.x_train.columns.tolist())):
            res=[]
            for j in numbers:
                self.df[i][0] = j
                res.append(self.model)
            full.append(res)
        full = np.array(full)
        full = full.T
        prob_data = pd.DataFrame(full , columns = self.x_train.columns.tolist())
        prob_data['number'] = numbers
        m_prob = []
    
        for i in self.x_train.columns.tolist():
            m_prob_data = abs((prob_data[i][counter1]-prob_data[i][counter2])/(numbers[1]-numbers[0]))
            m_prob.append(m_prob_data)

        data = ({'column name': self.x_train.columns.tolist() , 'slope of line' : m_prob})
        df_prob = pd.DataFrame(data)
        df_probability = df_prob.sort_values(by=['slope of line'], ascending=False)
        df_probability = df_probability.reset_index(drop=True)
        df_probability.to_csv('./result/Sensitivity_Analysis.csv')

#----------------------------------------------------------------------------------------------------------------#