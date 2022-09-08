import pandas as pd
import sklearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import logging
from matplotlib import pyplot as plt
from psynlig import pca_residual_variance
import os

from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from PyALE import ale
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scikitplot.helpers import binary_ks_curve
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import fbeta_score, matthews_corrcoef, average_precision_score
from sklearn.metrics import log_loss, brier_score_loss, top_k_accuracy_score
from sklearn.metrics import brier_score_loss, jaccard_score
from imblearn.over_sampling import SMOTE 
from tabulate import tabulate
from sklearn.tree import export_graphviz
from scipy.stats import pearsonr
import graphviz
from sklearn import preprocessing
import statistics
import pickle
import patsy 
import math
from sklearn.metrics.cluster import homogeneity_score
from tabulate import tabulate
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import balanced_accuracy_score


try :
    os.mkdir("./result")
except :
    pass

class finra:     
    def __init__(self,df,train_split_index,labels_column,model):
        self.x_train = df.iloc[:train_split_index, :].drop(columns = labels_column).reset_index(drop=True)
        self.y_train = df.iloc[:train_split_index, :][[labels_column]].reset_index(drop=True)
        self.x_test = df.iloc[train_split_index : , :].drop(columns = labels_column).reset_index(drop=True)
        self.y_test = df.iloc[train_split_index: , :][[labels_column]].reset_index(drop=True)
        self.df = df
        self.model = model

    def pca(self):
        plt.style.use('dark_background')
        pca = PCA()
        pca.fit_transform(self.x_train)
        pca_residual_variance(pca, marker='o', markersize=3, color = "yellow", alpha=0.8)
        plt.xlabel("# reduced components")
        plt.title("Residual variance -> Dimensionality Reduction with PCA")
        plt.savefig("./result/pca_plot.jpg")
 

    def pireson_correlation(self):
        def pearson_correlation1():
            for i in range(len(self.x_train.columns.tolist())):
                for j in range (i+1,len(self.x_train.columns.tolist())):
                    crr1 = self.x_train[self.x_train.columns[i]].tolist()
                    crr2 = self.x_train[self.x_train.columns[j]].tolist()
                    corr, _ = pearsonr(crr1, crr2)
                    corr = abs(corr)
                    if corr>=0.8:
                        yield self.x_train.columns[i]
                        yield self.x_train.columns[j]
                        yield corr

        high_correlation = []
        for x in pearson_correlation1():
            high_correlation.append(x)
        feature_1 = []
        feature_2 = []
        correlation_coefficient = []
        for i in range(int(len(high_correlation)/3)):
            feature_1.append(high_correlation[i*3])
            feature_2.append(high_correlation[i*3+1])
            correlation_coefficient.append(high_correlation[i*3+2])
        prs_corr = pd.DataFrame({'feature 1' : feature_1 , 'feature 2' : feature_2 ,
                                 'pearson correlation coefficient' : correlation_coefficient})
        prs_corr = prs_corr.sort_values(by=['pearson correlation coefficient'], ascending=False)
        prs_corr.to_csv('./result/pireson_correlation.csv')



    def VIF(self):
        #gather features
        train_df = pd.concat([self.x_train, self.y_train], axis=1).reset_index(drop = True) 
        features = "+".join(self.x_train.columns.tolist())
        # get y and X dataframes based on this regression:
        y, X = dmatrices(f'{train_df.columns[-1]} ~' + features, self.df, return_type='dataframe')
        # For each X, calculate VIF and save in dataframe
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        vif = vif[['features','VIF Factor']].dropna()
        vif.round(1).to_csv('./result/VIF.csv')
        
    def eigen_vals(self):
        train_df = pd.concat([self.x_train, self.y_train], axis=1).reset_index(drop = True) 
        eigen_vals_list = []
        for i in range (len(self.x_train.columns.tolist())): 
            x = np.asarray(patsy.dmatrix(f"{self.x_train.columns.tolist()[i]} + C({train_df.columns[-1]})", data = self.df))
            _, sing_as, _ = np.linalg.svd(x)
            sing_as = sing_as.tolist()
            eigen_vals_list.append(sing_as)
        eigen_vals_list = np.array(eigen_vals_list)
        eigen_vals_columns = [ '0' , '1' , '2']
        eigen_vals_index = self.x_train.columns.tolist()
        eigen_vals = pd.DataFrame(data = eigen_vals_list , index = eigen_vals_index , columns = eigen_vals_columns)
        eigen_vals.to_csv('./result/eigen_vals.csv')
        
    def homogeneity_corr(self):
        def homogeneity():
            for i in range(len(self.x_train.columns.tolist())):
                for j in range (i+1,len(self.x_train.columns.tolist())):
                    crr1 = self.x_train[self.x_train.columns[i]].tolist()
                    crr2 = self.x_train[self.x_train.columns[j]].tolist()
                    corr = homogeneity_score(crr1, crr2)
                    corr = abs(corr)
                    if corr>=0.9:
                        yield self.x_train.columns[i]
                        yield self.x_train.columns[j]
                        yield corr

        high_correlation = []
        for x in homogeneity():
            high_correlation.append(x)
        feature_1 = []
        feature_2 = []
        correlation_coefficient = []
        for i in range(int(len(high_correlation)/3)):
            feature_1.append(high_correlation[i*3])
            feature_2.append(high_correlation[i*3+1])
            correlation_coefficient.append(high_correlation[i*3+2])
        homogeneity_corr = pd.DataFrame({'feature 1' : feature_1 , 'feature 2' : feature_2 ,
                                 'homogeneity correlation coefficient' : correlation_coefficient})
        homogeneity_corr = homogeneity_corr.sort_values(by=['homogeneity correlation coefficient'], ascending=False).reset_index(drop=True)
        homogeneity_corr.to_csv('./result/Homogeneity_Correlation.csv')

    
    def Balance_check(self):
        train_df = pd.concat([self.x_train, self.y_train], axis=1).reset_index(drop = True) 
        test_df = pd.concat([self.x_test, self.y_test], axis=1).reset_index(drop = True)
        train_frauds = (sum(train_df[train_df.columns[-1]])/len(train_df))*100
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

        x_all = pd.concat([self.x_train, self.x_test], axis=0).reset_index(drop = True)
        cluster = 10
        data = np.array(x_all).reshape(-1, x_all.shape[1])
        x_all['cluster_group'] = self.model.fit_predict(data).tolist()
        train_df['cluster_group'] = x_all['cluster_group'][:len(self.x_train)].reset_index(drop = True)
        test_df['cluster_group'] = x_all['cluster_group'][len(self.x_train):].reset_index(drop = True)
        train_fraud = []
        test_fraud = []
        for i in range(cluster):
            df_1 = train_df[train_df['cluster_group']==i]
            train_fraud.append(round(100*sum(df_1[train_df.columns[-1]])/sum(train_df[train_df.columns[-1]]),2))
        for i in range(cluster):
            df_1 = test_df[test_df['cluster_group']==i]
            test_fraud.append(round(100*sum(df_1[train_df.columns[-1]])/sum(test_df[train_df.columns[-1]]),2))
        fig, ax = plt.subplots(figsize = (10, 5))
        index = np.arange(10)
        bar_width = 0.35
        rects1 = plt.bar(index, train_fraud, bar_width, color='orange', label='train')
        rects2 = plt.bar(index + bar_width, test_fraud, bar_width, color='navy',label='test')
        plt.xlabel('clusters')
        plt.ylabel('% frauds ')
        plt.title('Frauds in train&test clusters')
        plt.xticks(index + 0.5*bar_width, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/Frauds_in_train&test_clusters_plot1.jpg") 

        plt.clf()
        plt.plot(index, train_fraud, color = "yellow", label='train')
        plt.plot(index, test_fraud, color = "red", label='test')
        plt.xlabel('clusters')
        plt.ylabel('% frauds ')
        plt.title('Frauds in train&test clusters')
        plt.xticks(index , ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/Frauds_in_train&test_clusters_plot2.jpg") 

        tr_ts_df2 = pd.DataFrame({'% train_frauds' : train_fraud, '%test_frauds' : test_fraud})
        tr_ts_df2.to_csv('./result/%frauds_in_clusters.csv')


    def Error_analysis_train(self):
        #train step
        np.random.seed(31)
        svm_model = self.model 
        svm_model.fit(self.x_train)
        svm_pred = svm_model.predict(self.x_train)
        svm_pred = pd.Series(svm_pred).replace([-1,1],[1,0])
        svm_conf_mat = pd.DataFrame(confusion_matrix(self.y_train, svm_pred))
        #############
        self.svm_conf_mat = svm_conf_mat
        self.svm_pred = svm_pred
        #############

        x_all = pd.concat([self.x_train, self.x_test], axis=0).reset_index(drop = True)
        data = np.array(x_all).reshape(-1, x_all.shape[1])
        km = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
        x_all['cluster_group'] = km.fit_predict(data).tolist()
        train_df = pd.concat([self.x_train, self.y_train], axis=1).reset_index(drop = True)
        train_df['cluster_group'] = x_all['cluster_group'][:len(self.x_train)].reset_index(drop = True)
        cluster = 10
        index = []
        test_score = [] 
        tn, fp, fn, tp = confusion_matrix(self.y_train, svm_pred).ravel()
        index.append("accuracy")
        test_score.append((tp + tn) / (tp + fp + fn + tn))
        index.append("recall_score")
        test_score.append(recall_score(self.y_train, svm_pred)) 
        index.append("precision_score")
        test_score.append(precision_score(self.y_train, svm_pred))
        index.append("f1_score")
        test_score.append(f1_score(self.y_train, svm_pred)) 
        index.append("f2_score")
        test_score.append(fbeta_score(self.y_train, svm_pred, beta = 2))
        index.append("cohen_kappa_score")
        test_score.append(cohen_kappa_score(self.y_train, svm_pred))
        index.append("roc_auccuracy_score")
        test_score.append(roc_auc_score(self.y_train, svm_pred))
        index.append("false_positive_rate")
        test_score.append(fp / (fp + tn))
        index.append("false_negative_rate")
        test_score.append(fn / (tp + fn))
        index.append("true_negative_rate")
        test_score.append(tn / (tn + fp))
        index.append("negative_predictive_value")
        test_score.append(tn/ (tn + fn))
        index.append("false_discovery_rate")
        test_score.append(fp/ (tp + fp))
        index.append("matthews_corr")
        test_score.append(matthews_corrcoef(self.y_train, svm_pred))
        index.append("avg_precision")
        test_score.append(average_precision_score(self.y_train, svm_pred))
        index.append("log_loss")
        test_score.append(log_loss(self.y_train, svm_pred))
        index.append("brier_score_loss")
        test_score.append(brier_score_loss(self.y_train, svm_pred))
        index.append("binary_ks_curve")
        res = binary_ks_curve(self.y_train, svm_pred)
        test_score.append(res[3])        
        index.append("balanced_accuracy_score")
        test_score.append(balanced_accuracy_score(self.y_train, svm_pred))
        index.append("top_k_accuracy_score")
        test_score.append(top_k_accuracy_score(self.y_train, svm_pred, k=2))
        index.append("brier_score_loss")
        test_score.append(brier_score_loss(self.y_train, svm_pred))
        index.append("jaccard_score")
        test_score.append(jaccard_score(self.y_train, svm_pred))
        score_df1 = pd.DataFrame({"index":index ,"score" :test_score })
        y_df_train = pd.DataFrame(self.y_train).reset_index(drop = True)
        y_df_train['pred_label'] = svm_pred

        clusters = train_df.cluster_group.reset_index()
        y_df_train['cluster_group'] = clusters.cluster_group
        per_error_train = []
        for j in range(cluster):
            df4 = y_df_train[y_df_train.cluster_group == j]
            fr_df = df4[df4[train_df.columns[-2]] == 1]
            real = max(len(fr_df[train_df.columns[-2]]) , 1)
            err = len(fr_df[fr_df.pred_label == 0])
            per_error_train.append(100*err/real)
    
        self.score_df1 = score_df1
        self.per_error_train = per_error_train
        print(svm_pred)
    def Error_analysis_test(self):
        svm_model = self.model
        svm_model.fit(self.x_test)
        svm_pred_test = svm_model.predict(self.x_test)
        svm_pred_test = pd.Series(svm_pred_test).replace([-1,1],[1,0])
        svm_conf_mat_test = pd.DataFrame(confusion_matrix(self.y_test, svm_pred_test))
        ################
        self.svm_conf_mat_test = svm_conf_mat_test
        self.svm_pred_test = svm_pred_test
        ################
        index = []
        test_score = []
        x_all = pd.concat([self.x_train, self.x_test], axis=0).reset_index(drop = True)
        data = np.array(x_all).reshape(-1, x_all.shape[1])
        km = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
        x_all['cluster_group'] = km.fit_predict(data).tolist()
        test_df = pd.concat([self.x_test, self.y_test], axis=1).reset_index(drop = True)
        test_df['cluster_group'] = x_all['cluster_group'][len(self.x_train):].reset_index(drop = True)
        cluster = 10 
        tn, fp, fn, tp = confusion_matrix(self.y_test, svm_pred_test).ravel()
        index.append("accuracy")
        test_score.append((tp + tn) / (tp + fp + fn + tn))
        index.append("recall_score")
        test_score.append(recall_score(self.y_test, svm_pred_test)) 
        index.append("precision_score")
        test_score.append(precision_score(self.y_test, svm_pred_test))
        index.append("f1_score")
        test_score.append(f1_score(self.y_test, svm_pred_test)) 
        index.append("f2_score")
        test_score.append(fbeta_score(self.y_test, svm_pred_test, beta = 2))
        index.append("cohen_kappa_score")
        test_score.append(cohen_kappa_score(self.y_test, svm_pred_test))
        index.append("roc_auccuracy_score")
        test_score.append(roc_auc_score(self.y_test, svm_pred_test))
        index.append("false_positive_rate")
        test_score.append(fp / (fp + tn))
        index.append("false_negative_rate")
        test_score.append(fn / (tp + fn))
        index.append("true_negative_rate")
        test_score.append(tn / (tn + fp))
        index.append("negative_predictive_value")
        test_score.append(tn/ (tn + fn))
        index.append("false_discovery_rate")
        test_score.append(fp/ (tp + fp))
        index.append("matthews_corr")
        test_score.append(matthews_corrcoef(self.y_test, svm_pred_test))
        index.append("avg_precision")
        test_score.append(average_precision_score(self.y_test, svm_pred_test))
        index.append("log_loss")
        test_score.append(log_loss(self.y_test, svm_pred_test))
        index.append("brier_score_loss")
        test_score.append(brier_score_loss(self.y_test, svm_pred_test))
        index.append("binary_ks_curve")
        res = binary_ks_curve(self.y_test, svm_pred_test)
        test_score.append(res[3])
        index.append("balanced_accuracy_score")
        test_score.append(balanced_accuracy_score(self.y_test, svm_pred_test))
        index.append("top_k_accuracy_score")
        test_score.append(top_k_accuracy_score(self.y_test, svm_pred_test, k=2))
        index.append("brier_score_loss")
        test_score.append(brier_score_loss(self.y_test, svm_pred_test))
        index.append("jaccard_score")
        test_score.append(jaccard_score(self.y_test, svm_pred_test))
        score_df2 = pd.DataFrame({"index":index ,"score" :test_score })
        y_df_test = pd.DataFrame(self.y_test).reset_index(drop = True)
        y_df_test['pred_label'] = svm_pred_test
        clusters = test_df.cluster_group.reset_index()
        y_df_test['cluster_group'] = clusters.cluster_group
        per_error_test = []
        for j in range(cluster):
            df5 = y_df_test[y_df_test.cluster_group == j]
            fr_df = df5[df5[test_df.columns[-2]] == 1]
            real = max(len(fr_df[test_df.columns[-2]]) , 1)
            err = len(fr_df[fr_df.pred_label == 0]) 
            per_error_test.append(100*err/real)
        self.score_df2 = score_df2
        self.per_error_test = per_error_test
    
    def confusion_matrix_train(self):
        self.Error_analysis_train()
        self.svm_conf_mat.to_csv('./result/confusion_matrix_train.csv')
        self.score_df1.to_csv('./result/score_train.csv') 

    def confusion_matrix_test(self):
        self.Error_analysis_test()
        self.svm_conf_mat_test.to_csv('./result/confusion_matrix_test.csv')
        self.score_df2.to_csv('./result/score_test.csv')   

    def false_negative(self):
        self.Error_analysis_test()
        self.Error_analysis_train()
        fig, ax = plt.subplots(figsize = (10, 5))
        index = np.arange(10)
        bar_width = 0.35
        rects1 = plt.bar(index, self.per_error_train, bar_width, color='orange', label='train')
        rects2 = plt.bar(index + bar_width, self.per_error_test, bar_width, color='navy',label='test')
        plt.xlabel('clusters')
        plt.ylabel('% # False Negative / # all_real_fraud ')
        plt.title(' % (False Negative /  all_real_fraud) in clusters')
        plt.xticks(index + 0.5*bar_width, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
        plt.legend()
        plt.tight_layout()
        plt.savefig("./result/false_negative_plot.jpg") 
        df_fn = pd.DataFrame({'train' : self.per_error_train, 'test' : self.per_error_test})
        df_fn.to_csv('./result/false_negative.csv')    

    def decision_tree(self): 
        self.Error_analysis_test()
        self.Error_analysis_train()
        train_df =  pd.concat([self.x_train.reset_index(drop =True), self.y_train.reset_index(drop =True)], axis=1)
        svm_df_train = pd.DataFrame(self.svm_pred, columns = ['pred_label']).reset_index(drop =True)
        train_df = pd.concat([train_df, svm_df_train], axis=1)
        test_df =  pd.concat([self.x_test.reset_index(drop =True), self.y_test.reset_index(drop =True)], axis=1)
        svm_df_test = pd.DataFrame(self.svm_pred_test, columns = ['pred_label']).reset_index(drop =True)
        test_df = pd.concat([test_df, svm_df_test], axis=1)
        new_df = pd.concat([train_df, test_df], axis=0).reset_index(drop =True)
        new_df['fn_error'] = None
        for r in range(len(new_df)) :
            if ((new_df[train_df.columns[-2]][r] == 1) and (new_df.pred_label[r] == 0)) :
                new_df.fn_error[r] = 1
            else :
                new_df.fn_error[r] = 0
        dt_df = new_df.drop([train_df.columns[-2], 'pred_label'], axis=1)

        x_new = dt_df.drop(['fn_error'], axis=1)
        y_new = dt_df.fn_error.astype('int')
        sm = SMOTE(random_state=42)
        x_blnc, y_blnc = sm.fit_resample(x_new, y_new)
        x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(x_train_dt, y_train_dt)
            score = dt_clf.score(x_test_dt, y_test_dt)
            test_accuracy.append(score)

        fig = plt.figure()
        ax = plt.axes()
        plt.plot(max_depth_range, test_accuracy, color = "yellow")
        plt.xlabel("max depth")
        plt.ylabel("test accuracy")
        plt.title("Tuning the depth of a tree")
        plt.savefig("./result/tuning_tree_plot.jpg")

        dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        dt_clf.fit(x_train_dt, y_train_dt)

        importances = pd.DataFrame({'feature':x_train_dt.columns,'importance':np.round(dt_clf.feature_importances_,3)})
        self.importances = importances.sort_values('importance',ascending=False)
        self.importances.iloc[:15, :].to_csv('./result/Feature_Importance.csv')  


        
        def tree_to_df(reg_tree, feature_names):
            tree_ = reg_tree.tree_
            feature_name = [
                feature_names[i] if i != sklearn.tree._tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]

            def recurse(node, row, ret):
                if tree_.feature[node] != sklearn.tree._tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    # Add rule to row and search left branch
                    row[-1].append(name + " <= " +  str(round(threshold,3)))
                    recurse(tree_.children_left[node], row, ret)
                    # Add rule to row and search right branch
                    row[-1].append(name + " > " +  str(round(threshold,3)))
                    recurse(tree_.children_right[node], row, ret)
                else:
                    # Add output rules and start a new row
                    label = tree_.value[node]
                    ret.append("Value: [" + str(label[0][0]) + ',' + str(label[0][1]) + "]")
                    row.append([])

            # Initialize
            rules = [[]]
            vals = []

            # Call recursive function with initial values
            recurse(0, rules, vals)

            # Convert to table and output
            df_tree = pd.DataFrame(rules).dropna(how='all')
            df_tree['Return'] = pd.Series(vals)
            columns = []
            for i in range(len(df_tree.columns)-1):
                columns.append(f"Depth {i}")
            columns.append('Values')
            df_tree.columns = columns
            return df_tree

        tree_to_df(dt_clf , self.x_train.columns.tolist()).to_csv('./result/decision_tree.csv')    

        self.x_train_dt = x_train_dt
        self.x_test_dt = x_test_dt
        self.y_train_dt = y_train_dt
        self.dt_clf = dt_clf
        
        
    def PDP(self):
        self.Error_analysis_test()
        self.Error_analysis_train()
        train_df =  pd.concat([self.x_train.reset_index(drop =True), self.y_train.reset_index(drop =True)], axis=1)
        svm_df_train = pd.DataFrame(self.svm_pred, columns = ['pred_label']).reset_index(drop =True)
        train_df = pd.concat([train_df, svm_df_train], axis=1)
        test_df =  pd.concat([self.x_test.reset_index(drop =True), self.y_test.reset_index(drop =True)], axis=1)
        svm_df_test = pd.DataFrame(self.svm_pred_test, columns = ['pred_label']).reset_index(drop =True)
        test_df = pd.concat([test_df, svm_df_test], axis=1)
        new_df = pd.concat([train_df, test_df], axis=0).reset_index(drop =True)
        new_df['fn_error'] = None
        for r in range(len(new_df)) :
            if ((new_df[train_df.columns[-2]][r] == 1) and (new_df.pred_label[r] == 0)) :
                new_df.fn_error[r] = 1
            else :
                new_df.fn_error[r] = 0
        dt_df = new_df.drop([train_df.columns[-2], 'pred_label'], axis=1)

        x_new = dt_df.drop(['fn_error'], axis=1)
        y_new = dt_df.fn_error.astype('int')
        sm = SMOTE(random_state=42)
        x_blnc, y_blnc = sm.fit_resample(x_new, y_new)
        x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(x_train_dt, y_train_dt)
            score = dt_clf.score(x_test_dt, y_test_dt)
            test_accuracy.append(score)

        dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        dt_clf.fit(x_train_dt, y_train_dt)
        importances = pd.DataFrame({'feature':x_train_dt.columns,'importance':np.round(dt_clf.feature_importances_,3)})
        importances = importances.sort_values('importance',ascending=False)
        features = importances.feature[:2].tolist()
        PartialDependenceDisplay.from_estimator(dt_clf,
                                       X = x_train_dt,
                                       features = features, 
                                       target=0)
        plt.savefig("./result/PDP_plot.jpg")
         

    def ICE(self):
        self.Error_analysis_test()
        self.Error_analysis_train()
        train_df =  pd.concat([self.x_train.reset_index(drop =True), self.y_train.reset_index(drop =True)], axis=1)
        svm_df_train = pd.DataFrame(self.svm_pred, columns = ['pred_label']).reset_index(drop =True)
        train_df = pd.concat([train_df, svm_df_train], axis=1)
        test_df =  pd.concat([self.x_test.reset_index(drop =True), self.y_test.reset_index(drop =True)], axis=1)
        svm_df_test = pd.DataFrame(self.svm_pred_test, columns = ['pred_label']).reset_index(drop =True)
        test_df = pd.concat([test_df, svm_df_test], axis=1)
        new_df = pd.concat([train_df, test_df], axis=0).reset_index(drop =True)
        new_df['fn_error'] = None
        for r in range(len(new_df)) :
            if ((new_df[train_df.columns[-2]][r] == 1) and (new_df.pred_label[r] == 0)) :
                new_df.fn_error[r] = 1
            else :
                new_df.fn_error[r] = 0
        dt_df = new_df.drop([train_df.columns[-2], 'pred_label'], axis=1)

        x_new = dt_df.drop(['fn_error'], axis=1)
        y_new = dt_df.fn_error.astype('int')
        sm = SMOTE(random_state=42)
        x_blnc, y_blnc = sm.fit_resample(x_new, y_new)
        x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(x_train_dt, y_train_dt)
            score = dt_clf.score(x_test_dt, y_test_dt)
            test_accuracy.append(score)

        dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        dt_clf.fit(x_train_dt, y_train_dt)
        importances = pd.DataFrame({'feature':x_train_dt.columns,'importance':np.round(dt_clf.feature_importances_,3)})
        importances = importances.sort_values('importance',ascending=False)
        features = importances.feature[:2].tolist()
        PartialDependenceDisplay.from_estimator(dt_clf, x_train_dt, features ,kind='individual')
        plt.savefig("./result/ICE_plot.jpg")
        

    def ALE(self):
        self.Error_analysis_test()
        self.Error_analysis_train()
        train_df =  pd.concat([self.x_train.reset_index(drop =True), self.y_train.reset_index(drop =True)], axis=1)
        svm_df_train = pd.DataFrame(self.svm_pred, columns = ['pred_label']).reset_index(drop =True)
        train_df = pd.concat([train_df, svm_df_train], axis=1)
        test_df =  pd.concat([self.x_test.reset_index(drop =True), self.y_test.reset_index(drop =True)], axis=1)
        svm_df_test = pd.DataFrame(self.svm_pred_test, columns = ['pred_label']).reset_index(drop =True)
        test_df = pd.concat([test_df, svm_df_test], axis=1)
        new_df = pd.concat([train_df, test_df], axis=0).reset_index(drop =True)
        new_df['fn_error'] = None
        for r in range(len(new_df)) :
            if ((new_df[train_df.columns[-2]][r] == 1) and (new_df.pred_label[r] == 0)) :
                new_df.fn_error[r] = 1
            else :
                new_df.fn_error[r] = 0
        dt_df = new_df.drop([train_df.columns[-2], 'pred_label'], axis=1)

        x_new = dt_df.drop(['fn_error'], axis=1)
        y_new = dt_df.fn_error.astype('int')
        sm = SMOTE(random_state=42)
        x_blnc, y_blnc = sm.fit_resample(x_new, y_new)
        x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(x_train_dt, y_train_dt)
            score = dt_clf.score(x_test_dt, y_test_dt)
            test_accuracy.append(score)

        dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        dt_clf.fit(x_train_dt, y_train_dt)
        importances = pd.DataFrame({'feature':x_train_dt.columns,'importance':np.round(dt_clf.feature_importances_,3)})
        importances = importances.sort_values('importance',ascending=False)
        features = importances.feature[:2].tolist()
        ale_eff = ale(X=x_train_dt, model=dt_clf, feature=[features[0]], grid_size=50, include_CI=True, C=0.95)
        plt.savefig("./result/ALE_1D_plot.jpg")
        ale_eff2D = ale(X=x_train_dt, model=dt_clf, feature=features, grid_size=100)
        plt.savefig("./result/ALE_2D_plot.jpg")

    def Permutation_feature_importance(self):
        self.Error_analysis_test()
        self.Error_analysis_train()
        train_df =  pd.concat([self.x_train.reset_index(drop =True), self.y_train.reset_index(drop =True)], axis=1)
        svm_df_train = pd.DataFrame(self.svm_pred, columns = ['pred_label']).reset_index(drop =True)
        train_df = pd.concat([train_df, svm_df_train], axis=1)
        test_df =  pd.concat([self.x_test.reset_index(drop =True), self.y_test.reset_index(drop =True)], axis=1)
        svm_df_test = pd.DataFrame(self.svm_pred_test, columns = ['pred_label']).reset_index(drop =True)
        test_df = pd.concat([test_df, svm_df_test], axis=1)
        new_df = pd.concat([train_df, test_df], axis=0).reset_index(drop =True)
        new_df['fn_error'] = None
        for r in range(len(new_df)) :
            if ((new_df[train_df.columns[-2]][r] == 1) and (new_df.pred_label[r] == 0)) :
                new_df.fn_error[r] = 1
            else :
                new_df.fn_error[r] = 0
        dt_df = new_df.drop([train_df.columns[-2], 'pred_label'], axis=1)

        x_new = dt_df.drop(['fn_error'], axis=1)
        y_new = dt_df.fn_error.astype('int')
        sm = SMOTE(random_state=42)
        x_blnc, y_blnc = sm.fit_resample(x_new, y_new)
        x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(x_train_dt, y_train_dt)
            score = dt_clf.score(x_test_dt, y_test_dt)
            test_accuracy.append(score)

        dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        dt_clf.fit(x_train_dt, y_train_dt)
        r = permutation_importance(dt_clf, x_train_dt, y_train_dt,
                           n_repeats=30,
                           random_state=0)
        a = []
        column_name = []
        column_value = []
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                a.append(i)
                column_name.append(f"{self.x_train.columns[i]:<8}")
                column_value.append(f"{r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}")
        Permutation_feature_importance_df = pd.DataFrame({'feature':column_name , 'value':column_value})
        Permutation_feature_importance_df.to_csv('./result/Permutation_feature_importance.csv') 
  
    def Global_Surrogate(self):
        self.Error_analysis_test()
        self.Error_analysis_train()
        train_df =  pd.concat([self.x_train.reset_index(drop =True), self.y_train.reset_index(drop =True)], axis=1)
        svm_df_train = pd.DataFrame(self.svm_pred, columns = ['pred_label']).reset_index(drop =True)
        train_df = pd.concat([train_df, svm_df_train], axis=1)
        test_df =  pd.concat([self.x_test.reset_index(drop =True), self.y_test.reset_index(drop =True)], axis=1)
        svm_df_test = pd.DataFrame(self.svm_pred_test, columns = ['pred_label']).reset_index(drop =True)
        test_df = pd.concat([test_df, svm_df_test], axis=1)
        new_df = pd.concat([train_df, test_df], axis=0).reset_index(drop =True)
        new_df['fn_error'] = None
        for r in range(len(new_df)) :
            if ((new_df[train_df.columns[-2]][r] == 1) and (new_df.pred_label[r] == 0)) :
                new_df.fn_error[r] = 1
            else :
                new_df.fn_error[r] = 0
        dt_df = new_df.drop([train_df.columns[-2], 'pred_label'], axis=1)

        x_new = dt_df.drop(['fn_error'], axis=1)
        y_new = dt_df.fn_error.astype('int')
        sm = SMOTE(random_state=42)
        x_blnc, y_blnc = sm.fit_resample(x_new, y_new)
        x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(x_train_dt, y_train_dt)
            score = dt_clf.score(x_test_dt, y_test_dt)
            test_accuracy.append(score)

        dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        dt_clf.fit(x_train_dt, y_train_dt)
        new_target = dt_clf.predict(x_train_dt)

        # defining the interpretable decision tree model
        dt_model1 = DecisionTreeRegressor(max_depth=5, random_state=10)

        # fitting the surrogate decision tree model using the training set and new target
        dt_model1.fit(x_train_dt,new_target)

    def LIME(self):
        self.Error_analysis_test()
        self.Error_analysis_train()
        train_df =  pd.concat([self.x_train.reset_index(drop =True), self.y_train.reset_index(drop =True)], axis=1)
        svm_df_train = pd.DataFrame(self.svm_pred, columns = ['pred_label']).reset_index(drop =True)
        train_df = pd.concat([train_df, svm_df_train], axis=1)
        test_df =  pd.concat([self.x_test.reset_index(drop =True), self.y_test.reset_index(drop =True)], axis=1)
        svm_df_test = pd.DataFrame(self.svm_pred_test, columns = ['pred_label']).reset_index(drop =True)
        test_df = pd.concat([test_df, svm_df_test], axis=1)
        new_df = pd.concat([train_df, test_df], axis=0).reset_index(drop =True)
        new_df['fn_error'] = None
        for r in range(len(new_df)) :
            if ((new_df[train_df.columns[-2]][r] == 1) and (new_df.pred_label[r] == 0)) :
                new_df.fn_error[r] = 1
            else :
                new_df.fn_error[r] = 0
        dt_df = new_df.drop([train_df.columns[-2], 'pred_label'], axis=1)

        x_new = dt_df.drop(['fn_error'], axis=1)
        y_new = dt_df.fn_error.astype('int')
        sm = SMOTE(random_state=42)
        x_blnc, y_blnc = sm.fit_resample(x_new, y_new)
        x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(x_train_dt, y_train_dt)
            score = dt_clf.score(x_test_dt, y_test_dt)
            test_accuracy.append(score)

        dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        dt_clf.fit(x_train_dt, y_train_dt)
        explainer = LimeTabularExplainer(x_train_dt.values, mode="regression", feature_names=x_train_dt.columns)
        r = permutation_importance(dt_clf, x_train_dt, y_train_dt,
                           n_repeats=30,
                           random_state=0)
        a = []
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                a.append(i)
        # storing a new observation
        i = 6
        feature_name = []
        feature_value = []
        for i in a:
            X_observation = x_test_dt.iloc[[i], :]
            feature_name.append(self.x_train.columns[i])
            feature_value.append(dt_clf.predict(X_observation)[0])
           
        RF_prediction_df = pd.DataFrame({'feature':feature_name , 'value':feature_value})
        RF_prediction_df.to_csv('./result/RF_prediction.csv')

        explanation = explainer.explain_instance(X_observation.values[0], dt_clf.predict)
        explanation.show_in_notebook(show_table=True, show_all=False)

    def Sensitivity_Analysis(self):

        interval= 40 #assumed
        numbers = [interval*float(x)/10 for x in range(-10 , 11)]

        if len(numbers)%2 == 0:
            counter1 = len(numbers)/2
            counter2 = counter1+1
        else:
            counter1 = (len(numbers)+1)/2
            counter2 = (len(numbers)-3)/2
        full=[]
        for i in range(len(self.x_train.columns.tolist())):
            res=[]
            for j in numbers:
                self.df[i][0] = j
                res.append(self.model)
            full.append(res)
        full = np.array(full)
        full = full.T
        prob_data = pd.DataFrame(full , columns = self.x_train.columns.tolist())
        prob_data['number'] = numbers
        m_prob = []
    
        for i in self.x_train.columns.tolist():
            m_prob_data = abs((prob_data[i][counter1]-prob_data[i][counter2])/(numbers[1]-numbers[0]))
            m_prob.append(m_prob_data)

        data = ({'column name': self.x_train.columns.tolist() , 'slope of line' : m_prob})
        df_prob = pd.DataFrame(data)
        df_probability = df_prob.sort_values(by=['slope of line'], ascending=False)
        df_probability = df_probability.reset_index(drop=True)
        df_probability.to_csv('./result/Sensitivity_Analysis.csv')

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