#----------------------------------------------------------------------------------------------------------------#
#from Validation_classes_main import Cluster
#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
#----------------------------------------------------------------------------------------------------------------#
from sklearn.metrics import recall_score, precision_score, cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import fbeta_score, matthews_corrcoef, average_precision_score, log_loss
from sklearn.metrics import brier_score_loss, top_k_accuracy_score, jaccard_score
from scikitplot.helpers import binary_ks_curve
#----------------------------------------------------------------------------------------------------------------#
class ModelAnalyzer:
    def __init__(self, data):
        self.per_error_train = None
        self.model_conf_mat_train = None
        self.tn_train = None
        self.fp_train = None
        self.fn_train = None
        self.tp_train = None
        self.per_error_test = None
        self.model_conf_mat_test = None
        self.tn_test = None
        self.fp_test = None
        self.fn_test = None
        self.tp_test = None
        self.data = data
        
        np.random.seed(31)
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
            model_pred = np.argmax(model_pred, axis=1) 
        model_conf_mat = pd.DataFrame(confusion_matrix(self.data.y_train, model_pred))

        self.model_conf_mat_train = model_conf_mat
        self.model_pred_train = model_pred

        #df_all = self.data.df
        #df_all['cluster_group'] = self.data.cluster_group
        x_df = pd.concat([self.data.x_train, self.data.y_train], axis=1).reset_index(drop = True)
        x_df['cluster_group'] = self.data.cluster_group[:len(self.data.x_train)]

        self.tn_train, self.fp_train, self.fn_train, self.tp_train = confusion_matrix(self.data.y_train, model_pred).ravel()
        y_df = pd.DataFrame(self.data.y_train).reset_index(drop = True)
        y_df['pred_label'] = self.model_pred_train
        clusters = x_df.cluster_group.reset_index()
        y_df['cluster_group'] = clusters.cluster_group
        per_error = []
        for j in range(self.data.optimum_number_of_cluster):
            df4 = y_df[y_df.cluster_group == j]
            fr_df = df4[df4[y_df.columns[0]] == 1]
            real = np.max([fr_df.shape[0] , 1])
            err = len(fr_df[fr_df.pred_label == 0])
            per_error.append(100*err/real)
    
        self.per_error_train = per_error
        
        model_pred = self.data.model.predict(self.data.x_test)
        #model_pred = pd.Series(model_pred).replace([-1,1],[1,0])
        if checker(model_pred):
            model_pred = np.argmax(model_pred, axis=1) 
        model_conf_mat = pd.DataFrame(confusion_matrix(self.data.y_test, model_pred))

        self.model_conf_mat_test = model_conf_mat
        self.model_pred_test = model_pred

        #df_all = self.data.df
        #df_all['cluster_group'] = self.data.cluster_group
        x_df = pd.concat([self.data.x_test, self.data.y_test], axis=1).reset_index(drop = True)
        x_df['cluster_group'] = self.data.cluster_group[len(self.data.x_train):]

        self.tn_test, self.fp_test, self.fn_test, self.tp_test = confusion_matrix(self.data.y_test, model_pred).ravel()
        y_df = pd.DataFrame(self.data.y_test).reset_index(drop = True)
        y_df['pred_label'] = self.model_pred_test
        clusters = x_df.cluster_group.reset_index()
        y_df['cluster_group'] = clusters.cluster_group
        per_error = []
        for j in range(self.data.optimum_number_of_cluster):
            df4 = y_df[y_df.cluster_group == j]
            fr_df = df4[df4[df4.columns[0]] == 1]
            real = np.max([fr_df.shape[0] , 1])
            err = len(fr_df[fr_df.pred_label == 0])
            per_error.append(100*err/real)
    
        self.per_error_test = per_error        
    
