#----------------------------------------------------------------------------------------------------------------#
from finra_classes_main import cluster
#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#----------------------------------------------------------------------------------------------------------------#

class balance(cluster):

    def __init__(self, df, train_split_index, labels_column, model):
        super().__init__(df, train_split_index, labels_column, model)   

#----------------------------------------------------------------------------------------------------------------#

    def Balance_check(self):
        self.number_of_cluster()
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

#----------------------------------------------------------------------------------------------------------------#