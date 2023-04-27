#----------------------------------------------------------------------------------------------------------------#
from ModelAnalyzer import ModelAnalyzer
#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
#----------------------------------------------------------------------------------------------------------------#
from sklearn.metrics import recall_score, precision_score, cohen_kappa_score, f1_score
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, jaccard_score
from sklearn.metrics import fbeta_score, roc_auc_score, average_precision_score
from sklearn.metrics import brier_score_loss, top_k_accuracy_score, log_loss
from scikitplot.helpers import binary_ks_curve
#----------------------------------------------------------------------------------------------------------------#

class ErrorAnalaysis:
    def __init__(self, data, analyzer = None):
        self.data = data
        if analyzer is None:
            self.model_analysis = ModelAnalyzer(self.data)
        else:
            self.model_analysis = analyzer

        self.model_pred_train = self.model_analysis.model_pred_train
        self.model_pred_test = self.model_analysis.model_pred_test

    def error_cluster(self):
        x_all = self.data.df
        x_all['cluster_group'] = self.data.cluster_group
        train_df = pd.concat([self.data.x_train, self.data.y_train], axis=1).reset_index(drop = True) 
        test_df = pd.concat([self.data.x_test, self.data.y_test], axis=1).reset_index(drop = True)
        train_df['cluster_group'] = x_all['cluster_group'][:len(self.data.x_train)].reset_index(drop = True)
        test_df['cluster_group'] = x_all['cluster_group'][len(self.data.x_train):].reset_index(drop = True)
        cluster = self.data.optimum_number_of_cluster
        train_fraud = []
        test_fraud = []

        for i in range(cluster):
            df_1 = train_df[train_df['cluster_group']==i]
            train_fraud.append(round(100*sum(df_1[train_df.columns[-1]])/sum(train_df[train_df.columns[-1]]),2))
        for i in range(cluster):
            df_1 = test_df[test_df['cluster_group']==i]
            test_fraud.append(round(100*sum(df_1[train_df.columns[-1]])/sum(test_df[train_df.columns[-1]]),2))
        fig, ax = plt.subplots(figsize = (10, 5))
        index = np.arange(cluster)
        bar_width = 0.35
        rects1 = plt.bar(index, train_fraud, bar_width, color='orange', label='train')
        rects2 = plt.bar(index + bar_width, test_fraud, bar_width, color='navy',label='test')
        plt.xlabel('clusters')
        plt.ylabel('% frauds ')
        plt.title('Frauds in train&test clusters')
        plt.xticks(index + 0.5*bar_width, list(range(self.data.optimum_number_of_cluster)))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/Frauds_in_train&test_clusters_plot1.jpg") 

        plt.clf()
        plt.plot(index, train_fraud, color = "yellow", label='train')
        plt.plot(index, test_fraud, color = "red", label='test')
        plt.xlabel('clusters')
        plt.ylabel('% frauds ')
        plt.title('Frauds in train&test clusters')
        plt.xticks(index , list(range(self.data.optimum_number_of_cluster)))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/Frauds_in_train&test_clusters_plot2.jpg") 
        tr_ts_df2 = pd.DataFrame({'% train_frauds' : train_fraud, '%test_frauds' : test_fraud})
        tr_ts_df2['number_of_sample_train'] = tr_ts_df2['% train_frauds'] * len(self.data.x_train)
        tr_ts_df2['number_of_sample_test'] = tr_ts_df2['%test_frauds'] * len(self.data.x_test)

        tr_ts_df2.to_csv('./result/%frauds_in_clusters.csv')
        return tr_ts_df2
        #print('-------- frauds in clusters --------')
        #print(tabulate(tr_ts_df2, headers = 'keys', tablefmt = 'psql'), '\n')

#----------------------------------------------------------------------------------------------------------------#

    def accuracy(self, analysis_type):
        return((getattr(self.model_analysis, 'tp_' + analysis_type) + getattr(self.model_analysis, 'tn_' + analysis_type)) / (getattr(self.model_analysis, 'tp_' + analysis_type) + getattr(self.model_analysis, 'fp_' + analysis_type) + getattr(self.model_analysis, 'tn_' + analysis_type) + getattr(self.model_analysis, 'fn_' + analysis_type)))

#----------------------------------------------------------------------------------------------------------------#

    def recall_score(self, analysis_type):
        return(recall_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type))) 

#----------------------------------------------------------------------------------------------------------------#
        
    def precision_score(self, analysis_type):
        return(precision_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))

#----------------------------------------------------------------------------------------------------------------#
        
    def f1_score(self, analysis_type):
        return(f1_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type))) 

#----------------------------------------------------------------------------------------------------------------#
        
    def f2_score(self, analysis_type):
        return (fbeta_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type), beta = 2))

#----------------------------------------------------------------------------------------------------------------#
        
    def cohen_kappa_score(self, analysis_type):
        return (cohen_kappa_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))

#----------------------------------------------------------------------------------------------------------------#
        
    def roc_auccuracy_score(self, analysis_type):
        return (roc_auc_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))

#----------------------------------------------------------------------------------------------------------------#
        
    def false_positive_rate(self, analysis_type):
        return (getattr(self.model_analysis, 'fp_' + analysis_type) / (getattr(self.model_analysis, 'tn_' + analysis_type) + getattr(self.model_analysis, 'fp_' + analysis_type)))

#----------------------------------------------------------------------------------------------------------------#
    
    def false_negative_rate(self, analysis_type):
        return (getattr(self.model_analysis, 'fn_' + analysis_type) / (getattr(self.model_analysis, 'tp_' + analysis_type) + getattr(self.model_analysis, 'fn_' + analysis_type)))

#----------------------------------------------------------------------------------------------------------------#
        
    def true_negative_rate(self, analysis_type):
        return (getattr(self.model_analysis, 'tn_' + analysis_type) / (getattr(self.model_analysis, 'tn_' + analysis_type) + getattr(self.model_analysis, 'fp_' + analysis_type)))

#----------------------------------------------------------------------------------------------------------------#
        
    def negative_predictive_value(self, analysis_type):
        return (getattr(self.model_analysis, 'tn_' + analysis_type) / (getattr(self.model_analysis, 'tn_' + analysis_type) + getattr(self.model_analysis, 'fn_' + analysis_type)))
 
#----------------------------------------------------------------------------------------------------------------#
       
    def false_discovery_rate(self, analysis_type):
        return (getattr(self.model_analysis, 'fp_' + analysis_type) / (getattr(self.model_analysis, 'tp_' + analysis_type) + getattr(self.model_analysis, 'fp_' + analysis_type)))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def matthews_corr(self, analysis_type):
        return (matthews_corrcoef(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def avg_precision(self, analysis_type):
        return (average_precision_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))
  
#----------------------------------------------------------------------------------------------------------------#
   
    def log_loss(self, analysis_type):
        return (log_loss(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))
 
#----------------------------------------------------------------------------------------------------------------#

    def brier_score_loss(self, analysis_type):
        return (brier_score_loss(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def binary_ks_curve(self, analysis_type):
        res = binary_ks_curve(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type))
        return (res[3])  
   
#----------------------------------------------------------------------------------------------------------------#
  
    def balanced_accuracy_score(self, analysis_type):
        return (balanced_accuracy_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def top_k_accuracy_score(self, analysis_type):
        return (top_k_accuracy_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type), k=2))
   
#----------------------------------------------------------------------------------------------------------------#
  
    def jaccard_score(self, analysis_type):
        return (jaccard_score(getattr(self.data, 'y_' + analysis_type), getattr(self.model_analysis, 'model_pred_' + analysis_type)))
 
#----------------------------------------------------------------------------------------------------------------#

    metrics_function_list = [accuracy,recall_score,precision_score,f1_score,f2_score,
        cohen_kappa_score,roc_auccuracy_score,false_positive_rate,false_negative_rate,
        true_negative_rate,negative_predictive_value,false_discovery_rate,matthews_corr,
        avg_precision,log_loss,brier_score_loss,binary_ks_curve,balanced_accuracy_score,
        top_k_accuracy_score,jaccard_score]

#----------------------------------------------------------------------------------------------------------------#
    
    def train_score(self):
        train_score = [func(self, 'train') for func in ErrorAnalaysis.metrics_function_list]
        index = [func.__name__ for func in ErrorAnalaysis.metrics_function_list]
        score_df1 = pd.DataFrame({"index":index ,"score" :train_score})
        self.model_analysis.model_conf_mat_train.to_csv('./result/confusion_matrix_train.csv')
        score_df1.to_csv('./result/score_train.csv')
        return score_df1
        #print('-------- train score --------')
        #print(tabulate(score_df1, headers = 'keys', tablefmt = 'psql'), '\n')
 
    def test_score(self):
        test_score = [func(self, 'test') for func in ErrorAnalaysis.metrics_function_test_list]
        index = [func.__name__ for func in ErrorAnalaysis.metrics_function_list]
        score_df2 = pd.DataFrame({"index":index ,"score" :test_score})
        self.model_analysis.model_conf_mat_test.to_csv('./result/confusion_matrix_test.csv')
        score_df2.to_csv('./result/score_test.csv')
        return score_df2
        #print('-------- test score --------')
        #print(tabulate(score_df2, headers = 'keys', tablefmt = 'psql'), '\n')

#----------------------------------------------------------------------------------------------------------------#

    def false_negative(self):
        
        fig, ax = plt.subplots(figsize = (10, 5))
        index = np.arange(self.data.optimum_number_of_cluster)
        bar_width = 0.35

        #self.Error_analysis(self.x_train, self.y_train)
        per_error_train = self.model_analysis.per_error_train
        rects1 = plt.bar(index, per_error_train, bar_width, color='orange', label='train')

        #self.Error_analysis(self.data.x_test, self.data.y_test)
        per_error_test = self.model_analysis.per_error_test
        rects2 = plt.bar(index + bar_width, per_error_test, bar_width, color='navy',label='test')

        plt.xlabel('clusters')
        plt.ylabel('% # False Negative / # all_real_fraud ')
        plt.title(' % (False Negative /  all_real_fraud) in clusters')
        plt.xticks(index + 0.5*bar_width, list(range(self.data.optimum_number_of_cluster)))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/false_negative_plot.jpg") 
        df_fn = pd.DataFrame({'train' : per_error_train, 'test' : per_error_test})
        df_fn.to_csv('./result/false_negative.csv')
        print('Fasle Negative Calculation Complete')
        print(tabulate(df_fn, headers = 'keys', tablefmt = 'psql'))
        return df_fn

#----------------------------------------------------------------------------------------------------------------#

    def per_cluster_analysis(self):

        x_test = self.data.x_test
        x_train = self.data.x_train
        y_train = x_train.target
        try:
            x_train = x_train.drop(['cluster','target'], axis = 1)
        except:
            pass
        
        self.data.model.fit(x_train,y_train)
        
        x_test['target'] = self.data.y_test
        x_train['target'] = self.data.y_train
        x_test['cluster'] = self.data.cluster_group[:len(self.data.x_test)]
        x_train['cluster'] = self.data.cluster_group[:len(self.data.x_train)]
        
        train_cluster_list, test_cluster_list = [], []
    
        for i in range(self.data.optimum_number_of_cluster):
            train_cluster_list.append(len(x_train[x_train.cluster == i])/len(x_train))
            test_cluster_list.append(len(x_test[x_test.cluster == i])/len(x_test))

        fig = plt.subplots(figsize =(12, 8))
        barWidth = 0.25

        br1 = np.arange(self.data.optimum_number_of_cluster)
        br2 = [x + barWidth for x in br1]

        plt.bar(br1, train_cluster_list, color ='r', width = barWidth,
            edgecolor ='black', label ='train')

        plt.bar(br2, test_cluster_list, color ='b', width = barWidth,
            edgecolor ='black', label ='test')

        plt.xlabel('clusters')
        plt.ylabel('samples per cluster')

        plt.legend()
        plt.xticks(br1 + 0.5*barWidth, list(range(self.data.optimum_number_of_cluster)))

        plt.tight_layout()
        #plt.savefig('./result/samples_per_cluster.png')

        roc_test = []
        precision_test = []
        recall_test = []
        roc_train = []
        precision_train = []
        recall_train = []        
        
        for i in range(self.data.optimum_number_of_cluster):
            xtemp_train = x_train[x_train.cluster == i].reset_index(drop = True)
            ytemp_train = xtemp_train.target
            xtemp_train = xtemp_train.drop(['target','cluster'], axis = 1)
            
            #print(self.data.model.predict(xtemp_train))
            try:
                model_predict_train = self.data.model.predict(xtemp_train)
            except:
                pass
            xtemp_test = x_test[x_test.cluster == i].reset_index(drop = True)
            ytemp_test = xtemp_test.target
            xtemp_test = xtemp_test.drop(['target','cluster'], axis = 1)
            try:
                model_predict_test = self.data.model.predict(xtemp_test)
            except:
                pass
            
            

            if self.data.analysis_type == 'binary':
                model_predict_test = np.argmax(model_predict_test, axis = 1)
                model_predict_train = np.argmax(model_predict_train, axis = 1)
            if len(set(model_predict_train)) != 1:
                if (len(ytemp_test) == len(model_predict_test)) and (len(ytemp_train) == len(model_predict_train)):
                    if len(np.unique(ytemp_test)) > 1:
                        roc_test.append(round(roc_auc_score(ytemp_test, model_predict_test)))
                        roc_train.append(round(roc_auc_score(ytemp_train, model_predict_train)))

                    else:
                        roc_test.append(1)
                        roc_train.append(1)

                precision_test.append(round(precision_score(ytemp_test, model_predict_test)))
                precision_train.append(round(precision_score(ytemp_train, model_predict_train)))

                recall_test.append(round(recall_score(ytemp_test, model_predict_test)))
                recall_train.append(round(recall_score(ytemp_train, model_predict_train)))
            
#----------------------------------------------------------------------------------------------------------------#        

        fig = plt.subplots(figsize =(12, 8))
        barWidth = 0.25

        br1 = np.arange(self.data.optimum_number_of_cluster)
        br2 = [x + barWidth for x in br1]

        plt.bar(br1, roc_train, color ='r', width = barWidth,
            edgecolor ='black', label ='train')

        plt.bar(br2, roc_test, color ='b', width = barWidth,
            edgecolor ='black', label ='test')

        plt.xlabel('clusters')
        plt.ylabel('ROC')

        plt.legend()
        plt.xticks(br1 + 0.5*barWidth, list(range(self.data.optimum_number_of_cluster)))

        plt.tight_layout()
        plt.savefig('./result/ROC_Cluster.png')

#----------------------------------------------------------------------------------------------------------------#        

        fig = plt.subplots(figsize =(12, 8))
        barWidth = 0.25

        br1 = np.arange(self.data.optimum_number_of_cluster)
        br2 = [x + barWidth for x in br1]

        plt.bar(br1, precision_train, color ='r', width = barWidth,
            edgecolor ='black', label ='train')

        plt.bar(br2, precision_test, color ='b', width = barWidth,
            edgecolor ='black', label ='test')

        plt.xlabel('clusters')
        plt.ylabel('Precision')

        plt.legend()
        plt.xticks(br1 + 0.5*barWidth, list(range(self.data.optimum_number_of_cluster)))

        plt.tight_layout()
        plt.savefig('./result/Precision_Cluster.png')

#----------------------------------------------------------------------------------------------------------------#

        fig = plt.subplots(figsize =(12, 8))
        barWidth = 0.25

        br1 = np.arange(self.data.optimum_number_of_cluster)
        br2 = [x + barWidth for x in br1]

        plt.bar(br1, recall_train, color ='r', width = barWidth,
            edgecolor ='black', label ='train')

        plt.bar(br2, recall_test, color ='b', width = barWidth,
            edgecolor ='black', label ='test')

        plt.xlabel('clusters')
        plt.ylabel('Recall')

        plt.legend()
        plt.xticks(br1 + 0.5*barWidth, list(range(self.data.optimum_number_of_cluster)))

        plt.tight_layout()
        plt.savefig('./result/Recall_Cluster.png')