#----------------------------------------------------------------------------------------------------------------#
from finra_classes_main import cluster
#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#----------------------------------------------------------------------------------------------------------------#
from sklearn.metrics import recall_score, precision_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import fbeta_score, matthews_corrcoef, average_precision_score
from sklearn.metrics import log_loss, brier_score_loss, top_k_accuracy_score
from sklearn.metrics import brier_score_loss, jaccard_score
from scikitplot.helpers import binary_ks_curve
#----------------------------------------------------------------------------------------------------------------#

class err_cluster(cluster):

    def __init__(self, df, train_split_index, labels_column, model):
        super().__init__(df, train_split_index, labels_column, model)

    ##############################################################################################################

    def error_cluster(self):
        self.number_of_cluster()
        self.x_all['cluster_group'] = self.cluster_group
        train_df = pd.concat([self.x_train, self.y_train], axis=1).reset_index(drop = True) 
        test_df = pd.concat([self.x_test, self.y_test], axis=1).reset_index(drop = True)
        train_df['cluster_group'] = self.x_all['cluster_group'][:len(self.x_train)].reset_index(drop = True)
        test_df['cluster_group'] = self.x_all['cluster_group'][len(self.x_train):].reset_index(drop = True)
        cluster = self.optimum_number_of_cluster
        train_fraud = []
        test_fraud = []
        for i in range(cluster):
            df_1 = train_df[train_df['cluster_group']==i]
            train_fraud.append(round(100*sum(df_1[self.labels_column])/sum(train_df[self.labels_column]),2))
        for i in range(cluster):
            df_1 = test_df[test_df['cluster_group']==i]
            test_fraud.append(round(100*sum(df_1[self.labels_column])/sum(test_df[self.labels_column]),2))
        fig, ax = plt.subplots(figsize = (10, 5))
        index = np.arange(10)
        bar_width = 0.35
        rects1 = plt.bar(index, train_fraud, bar_width, color='orange', label='train')
        rects2 = plt.bar(index + bar_width, test_fraud, bar_width, color='navy',label='test')
        plt.xlabel('clusters')
        plt.ylabel('% frauds ')
        plt.title('Frauds in train&test clusters')
        plt.xticks(index + 0.5*bar_width, list(range(self.optimum_number_of_cluster)))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/Frauds_in_train&test_clusters_plot1.jpg") 

        plt.clf()
        plt.plot(index, train_fraud, color = "yellow", label='train')
        plt.plot(index, test_fraud, color = "red", label='test')
        plt.xlabel('clusters')
        plt.ylabel('% frauds ')
        plt.title('Frauds in train&test clusters')
        plt.xticks(index , list(range(self.optimum_number_of_cluster)))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/Frauds_in_train&test_clusters_plot2.jpg") 

        tr_ts_df2 = pd.DataFrame({'% train_frauds' : train_fraud, '%test_frauds' : test_fraud})
        tr_ts_df2['number_of_sample_train'] = train_fraud * self.x_train
        tr_ts_df2['number_of_sample_test'] = test_fraud * self.x_test

        tr_ts_df2.to_csv('./result/%frauds_in_clusters.csv')

#----------------------------------------------------------------------------------------------------------------#
#-----------------------------                                                       ----------------------------#
#-----------------------------                      TRAIN STEP                       ----------------------------#
#-----------------------------                                                       ----------------------------#
#----------------------------------------------------------------------------------------------------------------#

    def Error_analysis_train(self):
        self.number_of_cluster()
        #train step
        np.random.seed(31)
        svm_model = self.model 
        svm_model.fit(self.x_train)
        svm_pred = svm_model.predict(self.x_train)
        svm_pred = pd.Series(svm_pred).replace([-1,1],[1,0])
        svm_conf_mat = pd.DataFrame(confusion_matrix(self.y_train, svm_pred))

        self.svm_conf_mat = svm_conf_mat
        self.svm_pred = svm_pred

        self.x_all['cluster_group'] = self.cluster_group
        train_df = pd.concat([self.x_train, self.y_train], axis=1).reset_index(drop = True)
        train_df['cluster_group'] = self.x_all['cluster_group'][:len(self.x_train)].reset_index(drop = True)

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_train, svm_pred).ravel()
        y_df_train = pd.DataFrame(self.y_train).reset_index(drop = True)
        y_df_train['pred_label'] = self.svm_pred

        clusters = train_df.cluster_group.reset_index()
        y_df_train['cluster_group'] = clusters.cluster_group
        per_error_train = []
        for j in range(self.optimum_number_of_cluster):
            df4 = y_df_train[y_df_train.cluster_group == j]
            fr_df = df4[df4[train_df.columns[-2]] == 1]
            real = max(len(fr_df[train_df.columns[-2]]) , 1)
            err = len(fr_df[fr_df.pred_label == 0])
            per_error_train.append(100*err/real)
    
        self.per_error_train = per_error_train

#----------------------------------------------------------------------------------------------------------------#

    def accuracy_train(self):
        self.Error_analysis_train()
        return((self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn))

#----------------------------------------------------------------------------------------------------------------#

    def recall_score_train(self):
        self.Error_analysis_train()
        return(recall_score(self.y_train, self.svm_pred)) 

#----------------------------------------------------------------------------------------------------------------#
        
    def precision_score_train(self):
        self.Error_analysis_train()
        return(precision_score(self.y_train, self.svm_pred))

#----------------------------------------------------------------------------------------------------------------#
        
    def f1_score_train(self):
        self.Error_analysis_train()
        return(f1_score(self.y_train, self.svm_pred)) 

#----------------------------------------------------------------------------------------------------------------#
        
    def f2_score_train(self):
        self.Error_analysis_train()
        return(fbeta_score(self.y_train, self.svm_pred, beta = 2))

#----------------------------------------------------------------------------------------------------------------#
        
    def cohen_kappa_score_train(self):
        self.Error_analysis_train()
        return(cohen_kappa_score(self.y_train, self.svm_pred))

#----------------------------------------------------------------------------------------------------------------#
        
    def roc_auccuracy_score_train(self):
        self.Error_analysis_train()
        return(roc_auc_score(self.y_train, self.svm_pred))

#----------------------------------------------------------------------------------------------------------------#
        
    def false_positive_rate_train(self):
        self.Error_analysis_train()
        return(self.fp / (self.fp + self.tn))

#----------------------------------------------------------------------------------------------------------------#
    
    def false_negative_rate_train(self):
        self.Error_analysis_train()
        return(self.fn / (self.tp + self.fn))

#----------------------------------------------------------------------------------------------------------------#
        
    def true_negative_rate_train(self):
        self.Error_analysis_train()
        return(self.tn / (self.tn + self.fp))

#----------------------------------------------------------------------------------------------------------------#
        
    def negative_predictive_value_train(self):
        self.Error_analysis_train()
        return(self.tn/ (self.tn + self.fn))
 
#----------------------------------------------------------------------------------------------------------------#
       
    def false_discovery_rate_train(self):
        self.Error_analysis_train()
        return(self.fp/ (self.tp + self.fp))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def matthews_corr_train(self):
        self.Error_analysis_train()
        return(matthews_corrcoef(self.y_train, self.svm_pred))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def avg_precision_train(self):
        self.Error_analysis_train()
        return(average_precision_score(self.y_train, self.svm_pred))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def log_loss_train(self):
        self.Error_analysis_train()
        return(log_loss(self.y_train, self.svm_pred))
 
#----------------------------------------------------------------------------------------------------------------#

    def brier_score_loss_train(self):
        self.Error_analysis_train()
        return(brier_score_loss(self.y_train, self.svm_pred))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def binary_ks_curve_train(self):
        self.Error_analysis_train()
        res = binary_ks_curve(self.y_train, self.svm_pred)
        return(res[3])  
   
#----------------------------------------------------------------------------------------------------------------#
  
    def balanced_accuracy_score_train(self):      
        return(balanced_accuracy_score(self.y_train, self.svm_pred))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def top_k_accuracy_score_train(self):
        self.Error_analysis_train()
        return(top_k_accuracy_score(self.y_train, self.svm_pred, k=2))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def jaccard_score_train(self):
        self.Error_analysis_train()
        return(jaccard_score(self.y_train, self.svm_pred))
 
#----------------------------------------------------------------------------------------------------------------#

    metrics_function_train_list = [accuracy_train,recall_score_train,precision_score_train,f1_score_train,f2_score_train,
        cohen_kappa_score_train,roc_auccuracy_score_train,false_positive_rate_train,false_negative_rate_train,
        true_negative_rate_train,negative_predictive_value_train,false_discovery_rate_train,matthews_corr_train,
        avg_precision_train,log_loss_train,brier_score_loss_train,binary_ks_curve_train,balanced_accuracy_score_train,
        top_k_accuracy_score_train,jaccard_score_train]
 
#----------------------------------------------------------------------------------------------------------------#

    def train_score(self):
        self.Error_analysis_train()
        train_score = [func(self) for func in err_cluster.metrics_function_train_list]
        index = [func.__name__ for func in err_cluster.metrics_function_train_list]
        score_df1 = pd.DataFrame({"index":index ,"score" :train_score })
        self.svm_conf_mat.to_csv('./result/confusion_matrix_train.csv')
        score_df1.to_csv('./result/score_test.csv')
 
#----------------------------------------------------------------------------------------------------------------#
#-----------------------------                                                       ----------------------------#
#-----------------------------                      TEST STEP                        ----------------------------#
#-----------------------------                                                       ----------------------------#
#----------------------------------------------------------------------------------------------------------------#

    def Error_analysis_test(self):
        self.number_of_cluster()
        svm_model = self.model
        svm_model.fit(self.x_test)
        svm_pred_test = svm_model.predict(self.x_test)
        svm_pred_test = pd.Series(svm_pred_test).replace([-1,1],[1,0])
        svm_conf_mat_test = pd.DataFrame(confusion_matrix(self.y_test, svm_pred_test))

        self.svm_conf_mat_test = svm_conf_mat_test
        self.svm_pred_test = svm_pred_test

        test_df = pd.concat([self.x_test, self.y_test], axis=1).reset_index(drop = True)
        test_df['cluster_group'] = self.x_all['cluster_group'][len(self.x_train):].reset_index(drop = True)
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_test, svm_pred_test).ravel()
        y_df_test = pd.DataFrame(self.y_test).reset_index(drop = True)
        y_df_test['pred_label'] = svm_pred_test
        clusters = test_df.cluster_group.reset_index()
        y_df_test['cluster_group'] = clusters.cluster_group
        per_error_test = []
        for j in range(self.optimum_number_of_cluster):
            df5 = y_df_test[y_df_test.cluster_group == j]
            fr_df = df5[df5[test_df.columns[-2]] == 1]
            real = max(len(fr_df[test_df.columns[-2]]) , 1)
            err = len(fr_df[fr_df.pred_label == 0]) 
            per_error_test.append(100*err/real)
        self.per_error_test = per_error_test
 
#----------------------------------------------------------------------------------------------------------------#

    def accuracy_test(self):
        self.Error_analysis_test()
        return((self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def recall_score_test(self):
        self.Error_analysis_test()
        return(recall_score(self.y_test, self.svm_pred_test)) 
   
#----------------------------------------------------------------------------------------------------------------#
  
    def precision_score_test(self):
        self.Error_analysis_test()
        return(precision_score(self.y_test, self.svm_pred_test))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def f1_score_test(self):
        self.Error_analysis_test()
        return(f1_score(self.y_test, self.svm_pred_test)) 
    
#----------------------------------------------------------------------------------------------------------------#
 
    def f2_score_test(self):
        self.Error_analysis_test()
        return(fbeta_score(self.y_test, self.svm_pred_test, beta = 2))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def cohen_kappa_score_test(self):
        self.Error_analysis_test()
        return(cohen_kappa_score(self.y_test, self.svm_pred_test))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def roc_auccuracy_score_test(self):
        self.Error_analysis_test()
        return(roc_auc_score(self.y_test, self.svm_pred_test))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def false_positive_rate_test(self):
        self.Error_analysis_test()
        return(self.fp / (self.fp + self.tn))
    
#----------------------------------------------------------------------------------------------------------------#
 
    def false_negative_rate_test(self):
        self.Error_analysis_test()
        return(self.fn / (self.tp + self.fn))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def true_negative_rate_test(self):
        self.Error_analysis_test()
        return(self.tn / (self.tn + self.fp))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def negative_predictive_value_test(self):
        self.Error_analysis_test()
        return(self.tn/ (self.tn + self.fn))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def false_discovery_rate_test(self):
        self.Error_analysis_test()
        return(self.fp/ (self.tp + self.fp))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def matthews_corr_test(self):
        self.Error_analysis_test()
        return(matthews_corrcoef(self.y_test, self.svm_pred_test))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def avg_precision_test(self):
        self.Error_analysis_test()
        return(average_precision_score(self.y_test, self.svm_pred_test))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def log_loss_test(self):
        self.Error_analysis_test()
        return(log_loss(self.y_test, self.svm_pred_test))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def brier_score_loss_test(self):
        self.Error_analysis_test()
        return(brier_score_loss(self.y_test, self.svm_pred_test))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def binary_ks_curve_test(self):
        self.Error_analysis_test()
        res = binary_ks_curve(self.y_test, self.svm_pred_test)
        return(res[3])  
  
#----------------------------------------------------------------------------------------------------------------#
   
    def balanced_accuracy_score_test(self):      
        return(balanced_accuracy_score(self.y_test, self.svm_pred_test))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def top_k_accuracy_score_test(self):
        self.Error_analysis_test()
        return(top_k_accuracy_score(self.y_test, self.svm_pred_test, k=2))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def jaccard_score_test(self):
        self.Error_analysis_test()
        return(jaccard_score(self.y_test, self.svm_pred_test))
 
#----------------------------------------------------------------------------------------------------------------#

    metrics_function_test_list = [accuracy_test,recall_score_test,precision_score_test,f1_score_test,f2_score_test,
        cohen_kappa_score_test,roc_auccuracy_score_test,false_positive_rate_test,false_negative_rate_test,
        true_negative_rate_test,negative_predictive_value_test,false_discovery_rate_test,matthews_corr_test,
        avg_precision_test,log_loss_test,brier_score_loss_test,binary_ks_curve_test,balanced_accuracy_score_test,
        top_k_accuracy_score_test,jaccard_score_test]

    def test_score(self):
        self.Error_analysis_test()
        test_score = [func(self) for func in err_cluster.metrics_function_test_list]
        index = [func.__name__ for func in err_cluster.metrics_function_test_list]
        score_df2 = pd.DataFrame({"index":index ,"score" :test_score })
        self.svm_conf_mat_test.to_csv('./result/confusion_matrix_test.csv')
        score_df2.to_csv('./result/score_test.csv')
 
#----------------------------------------------------------------------------------------------------------------#

    def false_negative(self):
        self.Error_analysis_test()
        self.Error_analysis_train()
        fig, ax = plt.subplots(figsize = (10, 5))
        index = np.arange(self.optimum_number_of_cluster)
        bar_width = 0.35
        rects1 = plt.bar(index, self.per_error_train, bar_width, color='orange', label='train')
        rects2 = plt.bar(index + bar_width, self.per_error_test, bar_width, color='navy',label='test')
        plt.xlabel('clusters')
        plt.ylabel('% # False Negative / # all_real_fraud ')
        plt.title(' % (False Negative /  all_real_fraud) in clusters')
        plt.xticks(index + 0.5*bar_width, list(range(self.optimum_number_of_cluster)))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/false_negative_plot.jpg") 
        df_fn = pd.DataFrame({'train' : self.per_error_train, 'test' : self.per_error_test})
        df_fn.to_csv('./result/false_negative.csv')

#----------------------------------------------------------------------------------------------------------------#