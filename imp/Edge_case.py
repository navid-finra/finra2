#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#

class  Edge_case:
    def __init__(self, data):
        
        self.data = data

    def edge_case_analysis(self):
        model = self.data.model
        x_train = self.data.x_train
        x_test = self.data.x_test

        binary_feats = [col for col in x_train if x_train[col].dropna().value_counts().index.isin([0,1]).all()]

        num_feats = x_train.drop(columns = binary_feats).columns
        #  epsilon for edge case
        eps = 0.05
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
        print(f'all_edge_cases = {round((sum(edge_df.edge_case_up) + sum(edge_df.edge_case_down)) / (x_test.shape[0] * x_test.shape[1]),3) } %')
        edge_df_pos = edge_df[edge_df.pred_label == 1]
        print(f'pos_edge_cases = {round((sum(edge_df_pos.edge_case_up) + sum(edge_df_pos.edge_case_down)) / ((edge_df_pos.shape[0] * edge_df_pos.shape[1])+0.01),3) } %')
        edge_df_neg = edge_df[edge_df.pred_label == 0]
        print(f'neg_edge_cases = {round((sum(edge_df_neg.edge_case_up) + sum(edge_df_neg.edge_case_down)) / ((edge_df_neg.shape[0] * edge_df_neg.shape[1])+0.01),3) } %\n\n\n')

        up_edge_cases_features = pd.DataFrame(edge_df[edge_df.edge_case_up == True].groupby("features").size(), columns = ['#']).sort_values('#', ascending=False)
        print(f'up_edge_cases_features : \n{up_edge_cases_features}\n\n\n')

        down_edge_cases_features = pd.DataFrame(edge_df[edge_df.edge_case_down == True].groupby("features").size(), columns = ['#']).sort_values('#', ascending=False)
        print(f'down_edge_cases_features : \n{down_edge_cases_features}')

        edge_df.to_csv('./edge_case_df.csv')
        up_edge_cases_features.to_csv('./up_edge_cases_features.csv')
        down_edge_cases_features.to_csv('./down_edge_cases_features.csv')

        print(f'all_edge_cases = {(sum(edge_df.edge_case_up) + sum(edge_df.edge_case_down)) / ((x_test.shape[0] * x_test.shape[1])+0.01) } %')
        print(f'pos_edge_cases = {(sum(edge_df_pos.edge_case_up) + sum(edge_df_pos.edge_case_down)) / ((edge_df_pos.shape[0] * edge_df_pos.shape[1])+0.01) } %')
        print(f'neg_edge_cases = {(sum(edge_df_neg.edge_case_up) + sum(edge_df_neg.edge_case_down)) / ((edge_df_neg.shape[0] * edge_df_neg.shape[1])+0.01) } %\n\n\n')
        return down_edge_cases_features, up_edge_cases_features
        

#----------------------------------------------------------------------------------------------------------------#