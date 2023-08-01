.#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#

class  Robustness:
    def __init__(self, data):
        self.data = data

    def robustness_analysis(self):
        try:
            num_feats = self.data.x_test.columns
        except:
            num_feats = pd.DataFrame(self.data.x_test).columns

        up = self.data.x_test.copy()
        down = self.data.x_test.copy()
        y_predict = self.data.model.predict(self.data.x_test).reshape(1,-1)[0]
        y_predict = np.array([int(round(x)) for x in y_predict])
        mean_x = self.data.x_test.mean()/100
        zero_mean = pd.DataFrame((mean_x == 0), columns = ['bool'])
        zero_mean = zero_mean.index[zero_mean['bool']==True].tolist()

        for j in range(1,101):
            for i in num_feats:
                if i in zero_mean:
                    up[i] += j*(0.01)
                    down[i] -= j*(0.01)
                else:
                    up[i] += mean_x[i] * j
                    down[i] -= mean_x[i] * j
                    
                    
                if up.shape[0] == 0:
                    break
            
                else:
                    up_pre = self.data.model.predict(up).reshape(1,-1)[0]
                    up_pre = np.array([int(round(x)) for x in up_pre])
                    up_arr = (up_pre == y_predict) + 0
                    down_pre = self.data.model.predict(down).reshape(1,-1)[0]
                    down_pre = np.array([int(round(x)) for x in down_pre])
                    down_arr = (down_pre == y_predict) + 0
                    up['y_pred'] = y_predict
                    down['y_pred'] = y_predict
                    up['predict'] = down_arr + up_arr
                    down['predict'] = down_arr + up_arr
                    up = up[up['predict'] == 2]
                    down = down[down['predict'] == 2]
                    
                    y_predict = up['y_pred']
                    up = up.drop(['y_pred','predict'], axis = 1)
                    down = down.drop(['y_pred','predict'], axis = 1)
        
            else:
        
                continue
            
            break
                
            up -= mean_x * j
            down += mean_x * j
        
            
        print('Robustness =', round(up.shape[0]/X_test.shape[0],2),'%')
#----------------------------------------------------------------------------------------------------------------#
