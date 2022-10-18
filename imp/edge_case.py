from finra_classes_main import finra    
import logging
import pandas as pd
    
class  edge_case(finra):
    def __init__(self, df, train_split_index, labels_column, model):
        super().__init__(df, train_split_index, labels_column, model)   

    def edge_case_analysis(self):
        model = self.model
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
        clf = model.fit(x_train, y_train)
        binary_feats = [col for col in x_train if 
                    x_train[col].dropna().value_counts().index.isin([0,1]).all()]
        num_feats = x_train.drop(columns = binary_feats).columns
        #  epsilon for h case
        eps = 0.1
        index_id = []
        feat_edge_index = []
        edge_case_sit = []
        for r in range(len(x_test)):
            real_label = clf.predict(x_test.iloc[r:r+1, :])[0]
            for c in range(len(num_feats)):
                # value + epsilon
                x_up = x_test.copy()
                x_up.loc[r, num_feats[c]] = x_up.loc[r, num_feats[c]] +  x_up.loc[r, num_feats[c]] * eps
                pred_up = clf.predict(x_up.iloc[r:r+1, :])[0]
                if pred_up == real_label :
                    pass
                else :
                    index_id.append(r)
                    feat_edge_index.append(num_feats[c])
                    edge_case_sit.append("up")
                # value - epsilon
                x_dwn = x_test.copy()
                x_dwn.loc[r, num_feats[c]] = x_dwn.loc[r, num_feats[c]] -  x_dwn.loc[r, num_feats[c]] * eps
                pred_dwn = clf.predict(x_dwn.iloc[r:r+1, :])[0]
                if pred_dwn == real_label :
                    pass
                else :
                    index_id.append(r)
                    feat_edge_index.append(num_feats[c])
                    edge_case_sit.append("down")
        edge_case_df = pd.DataFrame({"index_id" : index_id, "features" : feat_edge_index, "h_case" : edge_case_sit})
        edge_case_df.to_csv("./result/edge_case_df.csv")
        logging.debug(f'edge_case_percent = {len(edge_case_df) / (x_test.shape[0] * x_test.shape[1])}')
        logging.debug('-------------------------------------------------------------------------------')