#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#

class  Edge_case:
    def __init__(self, data):
        
        self.data = data

    def edge_case_analysis(self):
        epsilon_up_list = []
        epsilon_down_list = []
        data_mean = round(self.data.x_test.mean(),2)/100
        y_predict = self.data.model.predict(self.data.x_test)

        temp_up = 0
        temp_down = 0

        for j in range(1,101):
            x_epsilon = self.data.x_test.copy()
            epsilon_up = x_epsilon + (j * data_mean)
            epsilon_down = x_epsilon - (j * data_mean)
            y_up = self.data.model.predict(epsilon_up)
            y_down = self.data.model.predict(epsilon_down)
            df_up = pd.DataFrame(data = {'y_up':y_up , 'y_pred':y_predict})
            df_down = pd.DataFrame(data = {'y_down':y_down , 'y_pred':y_predict})
            diff_up = df_up[df_up.y_up != df_up.y_pred].shape[0]
            diff_down = df_down[df_down.y_down != df_down.y_pred].shape[0]
            epsilon_up_list.append(diff_up - temp_up)
            epsilon_down_list.append(diff_down - temp_down)
            temp_up = diff_up
            temp_down = diff_down

        print('pos_edge_cases =', round((self.data.x_test.shape[0] - sum(epsilon_up_list))/(self.data.x_test.shape[0]),2))
        print('neg_edge_cases =', round((self.data.x_test.shape[0] - sum(epsilon_down_list))/(self.data.x_test.shape[0]),2), '\n\n\n')
        

#----------------------------------------------------------------------------------------------------------------#