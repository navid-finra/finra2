#----------------------------------------------------------------------------------------------------------------#
import logging
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#

class  edge_case:
    def __init__(self,x_train,x_test,y_train,y_test,model):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model 

    def edge_case_analysis(self):
        model = self.model
        x_train = self.x_train
        x_test = self.x_test

        binary_feats = [col for col in x_train if 
                    x_train[col].dropna().value_counts().index.isin([0,1]).all()]
        num_feats = x_train.drop(columns = binary_feats).columns
        #  epsilon for edge case
        eps = 0.1
        edge_df = pd.DataFrame()
        for c in range(len(num_feats)):    
            pred_label = model.predict(x_test).tolist()

            # value + epsilon
            x_up = x_test.copy()
            x_up.loc[:, num_feats[c]] = x_up.loc[:, num_feats[c]] +  x_up.loc[:, num_feats[c]] * eps
            pred_up = model.predict(x_up).tolist()

            # value - epsilon
            x_dwn = x_test.copy()
            x_dwn.loc[:, num_feats[c]] = x_dwn.loc[:, num_feats[c]] -  x_dwn.loc[:, num_feats[c]] * eps
            pred_dwn = model.predict(x_dwn).tolist()

            df_ = pd.DataFrame({ 'pred_label' : pred_label, 'pred_label_up' : pred_up, 'pred_label_down' : pred_dwn })
            df_['features'] = num_feats[c]
            edge_df = pd.concat([edge_df, df_], axis = 0)

        edge_df['edge_case_up'] = edge_df['pred_label'] != edge_df['pred_label_up']
        edge_df['edge_case_down'] = edge_df['pred_label'] != edge_df['pred_label_down']
        print(f'all_edge_cases = {(sum(edge_df.edge_case_up) + sum(edge_df.edge_case_down)) / (x_test.shape[0] * x_test.shape[1]) } %')
        edge_df_pos = edge_df[edge_df.pred_label == 1]
        print(f'pos_edge_cases = {(sum(edge_df_pos.edge_case_up) + sum(edge_df_pos.edge_case_down)) / (edge_df_pos.shape[0] * edge_df_pos.shape[1]) } %')
        edge_df_neg = edge_df[edge_df.pred_label == 0]
        print(f'neg_edge_cases = {(sum(edge_df_neg.edge_case_up) + sum(edge_df_neg.edge_case_down)) / (edge_df_neg.shape[0] * edge_df_neg.shape[1]) } %\n\n\n')

        up_edge_cases_features = pd.DataFrame(edge_df[edge_df.edge_case_up == True].groupby("features").size(), columns = ['#']).sort_values('#', ascending=False)
        print(f'up_edge_cases_features : \n{up_edge_cases_features}\n\n\n')

        down_edge_cases_features = pd.DataFrame(edge_df[edge_df.edge_case_down == True].groupby("features").size(), columns = ['#']).sort_values('#', ascending=False)
        print(f'down_edge_cases_features : \n{down_edge_cases_features}')

        edge_df.to_csv('./edge_case_df.csv')
        up_edge_cases_features.to_csv('./up_edge_cases_features.csv')
        down_edge_cases_features.to_csv('./down_edge_cases_features.csv')

        logging.debug(f'all_edge_cases = {(sum(edge_df.edge_case_up) + sum(edge_df.edge_case_down)) / (x_test.shape[0] * x_test.shape[1]) } %')
        logging.debug(f'pos_edge_cases = {(sum(edge_df_pos.edge_case_up) + sum(edge_df_pos.edge_case_down)) / (edge_df_pos.shape[0] * edge_df_pos.shape[1]) } %')
        logging.debug(f'neg_edge_cases = {(sum(edge_df_neg.edge_case_up) + sum(edge_df_neg.edge_case_down)) / (edge_df_neg.shape[0] * edge_df_neg.shape[1]) } %\n\n\n')
        logging.debug('-------------------------------------------------------------------------------')
        del edge_df, edge_df_pos, edge_df_neg, up_edge_cases_features, down_edge_cases_features

#----------------------------------------------------------------------------------------------------------------#