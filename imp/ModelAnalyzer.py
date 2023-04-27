#----------------------------------------------------------------------------------------------------------------#
#from Validation_classes_main import Cluster
#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
#----------------------------------------------------------------------------------------------------------------#
from sklearn.metrics import confusion_matrix  
#----------------------------------------------------------------------------------------------------------------#

class ModelAnalyzer:
    def __init__(self, data):
        self.per_error_train = None
        self.model_conf_mat_train = None
        self.tn_train = None
        self.fn_train = None
        self.tp_train = None
        self.fp_train = None
        self.per_error_test = None
        self.model_conf_mat_test = None
        self.tn_test = None
        self.fn_test = None
        self.tp_test = None
        self.fp_test = None
        self.data = data

#----------------------------------------------------------------------------------------------------------------#

        np.random.seed(31)
        self.data.model.fit(self.data.x_train, self.data.y_train)
        model_pred = self.data.model.predict(self.data.x_train)
        #model_pred = pd.Series(model_pred).replace([-1,1],[1,0])
        def checker(x):
            if len(x.shape) > 1:
                if x.shape[1] > 1:
                    return True
                else:
                    return False
            else:
                return False

        if checker(model_pred):
            model_pred = np.argmax(model_pred, axis = 1)
        model_conf_mat_train = pd.DataFrame(confusion_matrix(self.data.y_train, model_pred))

        self.model_conf_mat_train = model_conf_mat_train
        self.model_pred_train = model_pred

        x_df = pd.concat([self.data.x_train, self.data.y_train], axis=1).reset_index(drop = True)
        x_df['cluster_group'] = self.data.cluster_group[:len(self.data.x_train)]

        self.tn_train, self.fp_train, self.fn_train, self.tp_train = confusion_matrix(self.data.y_train, model_pred).ravel()
        y_df = pd.DataFrame(self.data.y_train).reset_index(drop = True)
        y_df['pred_label'] = self.model_pred_train
        clusters = x_df.cluster_group.reset_index()
        y_df['cluster_group'] = clusters.cluster_group
        per_error_train = []
        for j in range(self.data.optimum_number_of_cluster):
            df4 = y_df[y_df.cluster_group == j]
            fr_df = df4[df4[x_df.columns[-2]] == 1]
            real = max(len(fr_df[x_df.columns[-2]]) , 1)
            err = len(fr_df[fr_df.pred_label == 0])
            per_error_train.append(100*err/real)
    
        self.per_error_train = per_error_train

#----------------------------------------------------------------------------------------------------------------#

        model_pred = self.data.model.predict(self.data.x_test)

        if checker(model_pred):
            model_pred = np.argmax(model_pred, axis = 1)
        model_conf_mat_test = pd.DataFrame(confusion_matrix(self.data.y_test, model_pred))

        self.model_conf_mat_test = model_conf_mat_test
        self.model_pred_test = model_pred

        x_df = pd.concat([self.data.x_test, self.data.y_test], axis=1).reset_index(drop = True)
        x_df['cluster_group'] = self.data.cluster_group[:len(self.data.x_test)]

        self.tn_test, self.fp_test, self.fn_test, self.tp_test = confusion_matrix(self.data.y_test, model_pred).ravel()
        y_df = pd.DataFrame(self.data.y_test).reset_index(drop = True)
        y_df['pred_label'] = self.model_pred_test
        clusters = x_df.cluster_group.reset_index()
        y_df['cluster_group'] = clusters.cluster_group
        per_error_test = []
        for j in range(self.data.optimum_number_of_cluster):
            df4 = y_df[y_df.cluster_group == j]
            fr_df = df4[df4[x_df.columns[-2]] == 1]
            real = max(len(fr_df[x_df.columns[-2]]) , 1)
            err = len(fr_df[fr_df.pred_label == 0])
            per_error_test.append(100*err/real)
    
        self.per_error_test = per_error_test
#----------------------------------------------------------------------------------------------------------------#
