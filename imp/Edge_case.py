#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#

class  Edge_case:
    def __init__(self, data):
        self.data = data

    def edge_case_analysis(self):
        try:
            num_feats = self.data.x_test.columns
        except:
            num_feats = pd.DataFrame(self.data.x_test).columns

        up = self.data.x_test.copy()
        down = self.data.x_test.copy()
        y_predict = model.predict(self.data.x_test).reshape(1,-1)[0]
        y_predict = np.array([int(round(x)) for x in y_predict])
        mean_x = self.data.x_test.mean()/100

        for j in range(1,101):
            for i in num_feats:
                up[i] += mean_x[i] * j
                down[i] -= mean_x[i] * j
                up_pre = self.data.model.predict(up).reshape(1,-1)[0]
                up_pre = np.array([int(round(x)) for x in up_pre])
                up_arr = (up_pre == y_predict) + 0

                down_pre = self.data.model.predict(down).reshape(1,-1)[0]
                down_pre = np.array([int(round(x)) for x in down_pre])
                down_arr = (down_pre == y_predict) + 0

                up['y_pred'] = y_predict
                up['predict'] = down_arr + up_arr
                up = up[up['predict'] == 2]
                y_predict = up['y_pred']
                up = up.drop(['y_pred','predict'], axis = 1)

                down['y_pred'] = y_predict
                down['predict'] = down_arr + up_arr
                down = down[down['predict'] == 2]
                down = down.drop(['y_pred','predict'], axis = 1)
            up -= mean_x * j
            down += mean_x * j

        print('Robustness =', round(up.shape[0]/self.data.x_test.shape[0],2),'%')
#----------------------------------------------------------------------------------------------------------------#