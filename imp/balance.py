#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
#----------------------------------------------------------------------------------------------------------------#

class balance:
    def __init__(self,x_train,x_test,y_train,y_test,model):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

    def Balance_check(self):
        train_df = pd.concat([self.x_train, self.y_train], axis=1).reset_index(drop = True) 
        test_df = pd.concat([self.x_test, self.y_test], axis=1).reset_index(drop = True)
        train_frauds = (sum(train_df[self.labels_column])/len(train_df))*100
        train_norms = 100 - train_frauds
        test_frauds = (sum(test_df[test_df.columns[-1]])/len(test_df))*100
        test_norms = 100 - test_frauds
        train_per = (train_norms, train_frauds)
        test_per = (test_norms, test_frauds)
        fig, ax = plt.subplots(figsize = (10, 5))
        index = np.arange(2)
        bar_width = 0.35
        rects1 = plt.bar(index, train_per, bar_width, color='orange', label='train')
        rects2 = plt.bar(index + bar_width, test_per, bar_width, color='navy',label='test')
        plt.xlabel('labels')
        plt.ylabel('% each label')
        plt.title('Balance check')
        plt.xticks(index + 0.5*bar_width, ('0', '1'))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/balance_check_plot.jpg")         

        tr_ts_df = pd.DataFrame({'train' : train_per, 'test' : test_per})
        tr_ts_df.to_csv('./result/balance_check.csv')
        print(tabulate(tr_ts_df, headers = 'keys', tablefmt = 'psql'))

#----------------------------------------------------------------------------------------------------------------#