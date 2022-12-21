#----------------------------------------------------------------------------------------------------------------#
import os
import pandas as pd
import numpy as np
#----------------------------------------------------------------------------------------------------------------#
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import preprocessing
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
        if np.isnan(data_transformed).sum()>0:
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
                test_cluster = km.fit_predict(self.x_test).tolist()
                train_cluster = km.fit_predict(self.x_train).tolist()
                test_cluster_list = []
                train_cluster_list = []

                for i in range (optimum_number_of_cluster):
                    test_cluster_list.append(round(test_cluster.count(i)/len(self.x_test),2))
                    train_cluster_list.append(round(train_cluster.count(i)/len(self.x_train),2))

                cluster_df = pd.DataFrame({'train':train_cluster_list, 'test':test_cluster_list})
                cluster_df.to_csv('./result/Percentage_Per_Cluster.csv')
                
                return cluster_group
            except:
                print('The tolerance value is very low, please increase the tolerance')   


#----------------------------------------------------------------------------------------------------------------#