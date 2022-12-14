#----------------------------------------------------------------------------------------------------------------#
from finra_classes_main import finra    
import logging
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#

class  edge_case(finra):
    def __init__(self, df, train_split_index, labels_column, model):
        super().__init__(df, train_split_index, labels_column, model) 

    def edge_case_analysis(self):
        model = self.model
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
        binary_feats = [col for col in x_train if 
                    x_train[col].dropna().value_counts().index.isin([0,1]).all()]
        num_feats = x_train.drop(columns = binary_feats).columns
        #  epsilon for edge case
        eps = 0.1
        real_list = []
        pred_list_up = []
        pred_list_down = []
        for c in range(len(num_feats)): 
            real_label = model.predict(x_test).tolist()
            real_list += real_label
            # value + epsilon
            x_up = x_test.copy()
            x_up.loc[:, num_feats[c]] = x_up.loc[:, num_feats[c]] +  x_up.loc[:, num_feats[c]] * eps
            pred_up = model.predict(x_up).tolist()
            pred_list_up += pred_up 

            # value - epsilon
            x_dwn = x_test.copy()
            x_dwn.loc[:, num_feats[c]] = x_dwn.loc[:, num_feats[c]] -  x_dwn.loc[:, num_feats[c]] * eps
            pred_dwn = model.predict(x_dwn).tolist()
            pred_list_down += pred_dwn
        edge_df = pd.DataFrame({ 'real_label' : real_list, 'pred_label_up' : pred_list_up, 'pred_label_down' : pred_list_down })
        edge_df['edge_case_up'] = edge_df['real_label'] != edge_df['pred_label_up']
        edge_df['edge_case_down'] = edge_df['real_label'] != edge_df['pred_label_down']
        edge_df.to_csv("./result/edge_case_df.csv")
        print(f'%_edge_cases = {(sum(edge_df.edge_case_up) + sum(edge_df.edge_case_down)) / (x_test.shape[0] * x_test.shape[1]) } %')
        logging.debug(f'%_edge_cases = {(sum(edge_df.edge_case_up) + sum(edge_df.edge_case_down)) / (x_test.shape[0] * x_test.shape[1]) } %')
        logging.debug('-------------------------------------------------------------------------------')
        del edge_df, real_list, pred_list_up, pred_list_down, x_up, x_dwn, x_test

#----------------------------------------------------------------------------------------------------------------#