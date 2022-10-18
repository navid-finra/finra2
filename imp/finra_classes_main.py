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

    def __init__(self,df,train_split_index,labels_column,model):
        self.train_df = df[df['split']=='train']
        self.val_df = df[df['split']=='val']
        self.test_df = df[df['split']=='test']
        
        self.x_train = df.iloc[:train_split_index, :].drop(columns = labels_column).reset_index(drop=True)
        self.y_train = df.iloc[:train_split_index, :][[labels_column]].reset_index(drop=True)
        self.x_test = df.iloc[train_split_index : , :].drop(columns = labels_column).reset_index(drop=True)
        self.y_test = df.iloc[train_split_index: , :][[labels_column]].reset_index(drop=True)
        self.df = df
        self.model = model
        self.labels_column = labels_column


#----------------------------------------------------------------------------------------------------------------#

class cluster(finra):

    def __init__(self, df, train_split_index, labels_column, model):
        super().__init__(df, train_split_index, labels_column, model)

#----------------------------------------------------------------------------------------------------------------#

    def number_of_cluster(self):
        self.x_all = pd.concat([self.x_train, self.x_test], axis=0).reset_index(drop = True)
        data = np.array(self.x_all).reshape(-1, self.x_all.shape[1])
        mms = MinMaxScaler()
        mms.fit(data)
        data_transformed = mms.transform(data)
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
        self.optimum_number_of_cluster = normalize_slope.index([x for x in normalize_slope if x<0.02][0])
        km = KMeans(n_clusters = self.optimum_number_of_cluster, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
        self.cluster_group = km.fit_predict(data).tolist()

#----------------------------------------------------------------------------------------------------------------#