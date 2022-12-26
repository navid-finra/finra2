#----------------------------------------------------------------------------------------------------------------#
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------------------------------------------#
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import recall_score, precision_score, roc_auc_score
#----------------------------------------------------------------------------------------------------------------#

try :
    os.mkdir("./result")
except :
    pass

#----------------------------------------------------------------------------------------------------------------#

class Validation:
    def __init__(self,x_train,x_test,y_train,y_test,model):
        #self.train_df = df[df['split']=='train']
        #self.val_df = df[df['split']=='val']
        #self.test_df = df[df['split']=='test']
        
        #self.x_train = df.iloc[:train_split_index, :].drop(columns = labels_column).reset_index(drop=True)
        #self.y_train = df.iloc[:train_split_index, :][[labels_column]].reset_index(drop=True)
        #self.x_test = df.iloc[train_split_index : , :].drop(columns = labels_column).reset_index(drop=True)
        #self.y_test = df.iloc[train_split_index: , :][[labels_column]].reset_index(drop=True)
        #self.df = df
        #self.model = model
        #self.labels_column = labels_column
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model



#----------------------------------------------------------------------------------------------------------------#

class Cluster:
    def __init__(self,x_train,x_test,y_train,y_test,model):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model



    def number_of_cluster(self):
        df_all = pd.concat([self.x_train, self.x_test])
        data = np.array(df_all).reshape(-1, df_all.shape[1])
        mms = MinMaxScaler()
        mms.fit(data)
        data_transformed = mms.transform(data)
        if np.isnan(data_transformed).sum() > 0:
            print(f'the combine data contains {np.isnan(data_transformed).sum()} nan values')
        else:
            Sum_of_squared_distances = []
            K = range(1,15)
            for k in K:
                km = KMeans(n_clusters=k)
                km = km.fit(data_transformed)
                Sum_of_squared_distances.append(km.inertia_)
            slope = []
            for i in range(len(Sum_of_squared_distances)-1):
                slope.append(round(Sum_of_squared_distances[i]-Sum_of_squared_distances[i+1],2))
            normalize_slope = [round(x,2) for x in preprocessing.normalize([np.array(slope)])[0]]

            try:
                optimum_number_of_cluster = normalize_slope.index([x for x in normalize_slope if x<0.02][0])
                km = KMeans(n_clusters = optimum_number_of_cluster, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
                cluster_group = km.fit_predict(data).tolist()
                self.optimum = optimum_number_of_cluster
                return cluster_group
            except:
                print('The tolerance value is very low, please increase the tolerance')   


    def Percentage_Per_Cluster(self):
        self.number_of_cluster()

        km = KMeans(n_clusters = self.optimum, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
        test_cluster = km.fit_predict(self.x_test).tolist()
        train_cluster = km.fit_predict(self.x_train).tolist()
        test_cluster_list = []
        train_cluster_list = []

        for i in range(self.optimum):
            test_cluster_list.append(test_cluster.count(i))
            train_cluster_list.append(round(train_cluster.count(i)))

        X_test = self.x_test.copy()
        X_test['target'] = self.y_test
        X_test['cluster'] = test_cluster


        fig = plt.subplots(figsize =(12, 8))
        barWidth = 0.25

        br1 = np.arange(self.optimum)
        br2 = [x + barWidth for x in br1]

        plt.bar(br1, train_cluster_list, color ='r', width = barWidth,
            edgecolor ='black', label ='train')

        plt.bar(br2, test_cluster_list, color ='b', width = barWidth,
            edgecolor ='black', label ='test')

        plt.xlabel('clusters')
        plt.ylabel('samples per cluster')

        plt.legend()

        max_train_test = max(max(test_cluster_list), max(train_cluster_list))

        for i in range(self.optimum):
            xtemp = X_test[X_test.cluster == i].reset_index(drop = True)
            ytemp = xtemp.target
            xtemp = xtemp.drop(['target', 'cluster'], axis = 1)
            
            model_predict = self.model.predict(xtemp)
            
            plt.text(
                x = i + barWidth/2, y = (max_train_test*(1-0.4)), 
                s = f'roc_auc:{round(roc_auc_score(ytemp, model_predict),2)}',
                color = 'white', ha='center', fontsize = 15,
                bbox = dict(facecolor = 'g', alpha = 0.9)
                    )
            
            plt.text(
                x = i + barWidth/2, y = (max_train_test*(1-0.4))-(max_train_test*0.1), 
                s = f'precision:{round(precision_score(ytemp, model_predict),2)}',
                color = 'white', ha='center', fontsize = 15,
                bbox = dict(facecolor = 'g', alpha = 0.9)
                    )

            plt.text(
                x = i + barWidth/2, y = (max_train_test*(1-0.4))-(max_train_test*0.2), 
                s = f'recall:{round(recall_score(ytemp, model_predict),2)}',
                color = 'white', ha='center', fontsize = 15,
                bbox = dict(facecolor = 'g', alpha = 0.9)
                    )
            
            plt.text(
                x = i + barWidth, y = test_cluster_list[i] + (max_train_test*0.01), 
                s = test_cluster_list[i],
                color = 'black', ha='center', fontsize = 10
                    )
            
            plt.text(
                x = i, y = train_cluster_list[i] + (max_train_test*0.01), 
                s = train_cluster_list[i],
                color = 'black', ha='center', fontsize = 10
                    )
            
            plt.text(
                x = i + barWidth, y = test_cluster_list[i] - (max_train_test*0.03), 
                s = f'{round(test_cluster_list[i]/sum(test_cluster_list),2)}%',
                color = 'white', ha='center', fontsize = 10
                    )
            
            plt.text(
                x = i, y = train_cluster_list[i] - (max_train_test*0.03), 
                s = f'{round(train_cluster_list[i]/sum(train_cluster_list),2)}%',
                color = 'white', ha='center', fontsize = 10
                    )

        plt.xticks(br1 + 0.5*barWidth, list(range(self.optimum)))    
        
        plt.tight_layout()
        plt.savefig("./result/samples_per_cluster.png")

#----------------------------------------------------------------------------------------------------------------#