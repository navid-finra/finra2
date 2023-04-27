#----------------------------------------------------------------------------------------------------------------#
import os
import pandas as pd
import numpy as np
from copy import copy
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

def inplacer(method):
    def warp(self, *a, **k):
        inplace = k.pop('inplace', True)
        if inplace:
            return method(self, *a, **k)
        else:
            return method(copy(self), *a, **k(self))
    return warp

class Cluster:
    def __init__(self,data):     
        self.data = data


    @inplacer
    def number_of_cluster(self, telorance = 0.03):
        df_all = pd.concat([self.data.x_train, self.data.x_test])
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
                self.data.optimum_number_of_cluster = normalize_slope.index([x for x in normalize_slope if x < telorance][0])
                km = KMeans(n_clusters = self.data.optimum_number_of_cluster, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
                self.data.cluster_group = km.fit_predict(data).tolist()
                return self
            except:
                print('The tolerance value is very low, please increase the tolerance')     

#----------------------------------------------------------------------------------------------------------------#