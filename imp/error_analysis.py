#----------------------------------------------------------------------------------------------------------------#
from validation_classes_main import cluster
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

class err_cluster:

    def __init__(self,x_train,x_test,y_train,y_test,model):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

        cluster_class = cluster(x_train,y_train,x_test,y_test,model)

        self.cluster_group = cluster_class.number_of_cluster()
        self.optimum_number_of_cluster = max(self.cluster_group)

#----------------------------------------------------------------------------------------------------------------#

    def error_cluster(self):
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
        index = np.arange(cluster)
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
        tr_ts_df2['number_of_sample_train'] = tr_ts_df2['% train_frauds'] * len(self.x_train)
        tr_ts_df2['number_of_sample_test'] = tr_ts_df2['%test_frauds'] * len(self.x_test)

        tr_ts_df2.to_csv('./result/%frauds_in_clusters.csv')
        print('-------- frauds in clusters --------')
        print(tabulate(tr_ts_df2, headers = 'keys', tablefmt = 'psql'), '\n')

#----------------------------------------------------------------------------------------------------------------#

    def Error_analysis(self,x,y):
        np.random.seed(31)
        self.model.fit(x)
        model_pred = self.model.predict(x)
        model_pred = pd.Series(model_pred).replace([-1,1],[1,0])
        model_conf_mat = pd.DataFrame(confusion_matrix(y, model_pred))

        self.model_conf_mat = model_conf_mat
        self.model_pred = model_pred

        self.x_all['cluster_group'] = self.cluster_group
        x_df = pd.concat([x, y], axis=1).reset_index(drop = True)
        x_df['cluster_group'] = self.x_all['cluster_group'][:len(x)].reset_index(drop = True)

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y, model_pred).ravel()
        y_df = pd.DataFrame(y).reset_index(drop = True)
        y_df['pred_label'] = self.model_pred
        clusters = x_df.cluster_group.reset_index()
        y_df['cluster_group'] = clusters.cluster_group
        per_error = []
        for j in range(self.optimum_number_of_cluster):
            df4 = y_df[y_df.cluster_group == j]
            fr_df = df4[df4[x_df.columns[-2]] == 1]
            real = max(len(fr_df[x_df.columns[-2]]) , 1)
            err = len(fr_df[fr_df.pred_label == 0])
            per_error.append(100*err/real)
    
        self.per_error = per_error

#----------------------------------------------------------------------------------------------------------------#

    def accuracy(self,x,y):
        self.Error_analysis(x,y)
        return((self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn))

#----------------------------------------------------------------------------------------------------------------#

    def recall_score(self,x,y):
        self.Error_analysis(x,y)
        return(recall_score(y, self.model_pred)) 

#----------------------------------------------------------------------------------------------------------------#
        
    def precision_score(self,x,y):
        self.Error_analysis(x,y)
        return(precision_score(y, self.model_pred))

#----------------------------------------------------------------------------------------------------------------#
        
    def f1_score(self,x,y):
        self.Error_analysis(x,y)
        return(f1_score(y, self.model_pred)) 

#----------------------------------------------------------------------------------------------------------------#
        
    def f2_score(self,x,y):
        self.Error_analysis(x,y)
        return(fbeta_score(y, self.model_pred, beta = 2))

#----------------------------------------------------------------------------------------------------------------#
        
    def cohen_kappa_score(self,x,y):
        self.Error_analysis(x,y)
        return(cohen_kappa_score(y, self.model_pred))

#----------------------------------------------------------------------------------------------------------------#
        
    def roc_auccuracy_score(self,x,y):
        self.Error_analysis(x,y)
        return(roc_auc_score(y, self.model_pred))

#----------------------------------------------------------------------------------------------------------------#
        
    def false_positive_rate(self,x,y):
        self.Error_analysis(x,y)
        return(self.fp / (self.fp + self.tn))

#----------------------------------------------------------------------------------------------------------------#
    
    def false_negative_rate(self,x,y):
        self.Error_analysis(x,y)
        return(self.fn / (self.tp + self.fn))

#----------------------------------------------------------------------------------------------------------------#
        
    def true_negative_rate(self,x,y):
        self.Error_analysis(x,y)
        return(self.tn / (self.tn + self.fp))

#----------------------------------------------------------------------------------------------------------------#
        
    def negative_predictive_value(self,x,y):
        self.Error_analysis(x,y)
        return(self.tn/ (self.tn + self.fn))
 
#----------------------------------------------------------------------------------------------------------------#
       
    def false_discovery_rate(self,x,y):
        self.Error_analysis(x,y)
        return(self.fp/ (self.tp + self.fp))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def matthews_corr(self,x,y):
        self.Error_analysis(x,y)
        return(matthews_corrcoef(y, self.model_pred))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def avg_precision(self,x,y):
        self.Error_analysis(x,y)
        return(average_precision_score(y, self.model_pred))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def log_loss(self,x,y):
        self.Error_analysis(x,y)
        return(log_loss(y, self.model_pred))
 
#----------------------------------------------------------------------------------------------------------------#

    def brier_score_loss(self,x,y):
        self.Error_analysis(x,y)
        return(brier_score_loss(y, self.model_pred))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def binary_ks_curve(self,x,y):
        self.Error_analysis(x,y)
        res = binary_ks_curve(y, self.model_pred)
        return(res[3])  
   
#----------------------------------------------------------------------------------------------------------------#
  
    def balanced_accuracy_score(self,x,y):
        self.Error_analysis(x,y)      
        return(balanced_accuracy_score(y, self.model_pred))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def top_k_accuracy_score(self,x,y):
        self.Error_analysis(x,y)
        return(top_k_accuracy_score(y, self.model_pred, k=2))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def jaccard_score(self,x,y):
        self.Error_analysis(x,y)
        return(jaccard_score(y, self.model_pred))
 
#----------------------------------------------------------------------------------------------------------------#

    metrics_function_list = [accuracy,recall_score,precision_score,f1_score,f2_score,
        cohen_kappa_score,roc_auccuracy_score,false_positive_rate,false_negative_rate,
        true_negative_rate,negative_predictive_value,false_discovery_rate,matthews_corr,
        avg_precision,log_loss,brier_score_loss,binary_ks_curve,balanced_accuracy_score,
        top_k_accuracy_score,jaccard_score]
 
#----------------------------------------------------------------------------------------------------------------#
    
    def train_score(self):
        x = self.x_train
        y = self.y_train
        self.Error_analysis(x,y)
        train_score = [func(self,x,y) for func in err_cluster.metrics_function_list]
        index = [func.__name__ for func in err_cluster.metrics_function_list]
        score_df1 = pd.DataFrame({"index":index ,"score" :train_score })
        self.model_conf_mat.to_csv('./result/confusion_matrix_train.csv')
        score_df1.to_csv('./result/score_train.csv')
        print('-------- test score --------')
        print(tabulate(score_df1, headers = 'keys', tablefmt = 'psql'), '\n')
 
    def test_score(self):
        x = self.x_test
        y = self.y_test
        self.Error_analysis(x,y)
        test_score = [func(self,x,y) for func in err_cluster.metrics_function_test_list]
        index = [func.__name__ for func in err_cluster.metrics_function_test_list]
        score_df2 = pd.DataFrame({"index":index ,"score" :test_score })
        self.model_conf_mat.to_csv('./result/confusion_matrix_test.csv')
        score_df2.to_csv('./result/score_test.csv')
        print('-------- test score --------')
        print(tabulate(score_df2, headers = 'keys', tablefmt = 'psql'), '\n')
 
#----------------------------------------------------------------------------------------------------------------#

    def false_negative(self):
        
        fig, ax = plt.subplots(figsize = (10, 5))
        index = np.arange(self.optimum_number_of_cluster)
        bar_width = 0.35

        self.Error_analysis(self.x_train, self.y_train)
        per_error_train = self.per_error
        rects1 = plt.bar(index, per_error_train, bar_width, color='orange', label='train')

        self.Error_analysis(self.x_test, self.y_test)
        per_error_test = self.per_error
        rects2 = plt.bar(index + bar_width, per_error_test, bar_width, color='navy',label='test')

        plt.xlabel('clusters')
        plt.ylabel('% # False Negative / # all_real_fraud ')
        plt.title(' % (False Negative /  all_real_fraud) in clusters')
        plt.xticks(index + 0.5*bar_width, list(range(self.optimum_number_of_cluster)))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/false_negative_plot.jpg") 
        df_fn = pd.DataFrame({'train' : per_error_train, 'test' : per_error_test})
        df_fn.to_csv('./result/false_negative.csv')
        print('-------- false negative --------')
        print(tabulate(df_fn, headers = 'keys', tablefmt = 'psql'))

#----------------------------------------------------------------------------------------------------------------#