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

class finra:
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

class cluster:
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
                return cluster_group
            except:
                print('The tolerance value is very low, please increase the tolerance')   


    def df_info(self, df, labels_column) :
        
        ind = []
        val = []
        ind.append('features')
        val.append(df.drop(columns = labels_column).shape[1])
        ind.append('targets')
        val.append(df[labels_column].shape[1])
        ind.append('samples')
        val.append(len(df))
        ind.append('dimention')
        val.append(str(df.shape))
        ind.append('duplicated')
        val.append(df.duplicated().sum())
        ind.append('missing_values')
        val.append(df.isnull().sum().sum())
        ind.append('class_a_samples')
        val.append(df.groupby(labels_column).size()[0])
        ind.append('class_b_samples')
        val.append(df.groupby(labels_column).size()[1])
        ind.append('balance(class_a / class_b)')
        val.append(df.groupby(labels_column).size()[0] / df.groupby(labels_column).size()[1])
        rep_df = pd.DataFrame({"ind" : ind, "val" : val})
        return(rep_df)   
#----------------------------------------------------------------------------------------------------------------#