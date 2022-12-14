#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
#----------------------------------------------------------------------------------------------------------------#
from sklearn.decomposition import PCA
from psynlig import pca_residual_variance
from scipy.stats import pearsonr
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics.cluster import homogeneity_score
#----------------------------------------------------------------------------------------------------------------#

class Feature_Redundancy:
    def __init__(self,x_train,x_test,y_train,y_test,model):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.df = pd.concat([self.x_train, self.x_test])


  
    def pca(self):
        plt.style.use('dark_background')
        pca = PCA()
        pca.fit_transform(self.x_train)
        pca_residual_variance(pca, marker='o', markersize=3, color = "yellow", alpha=0.8)
        plt.xlabel("# reduced components")
        plt.title("Residual variance -> Dimensionality Reduction with PCA")
        plt.savefig("./result/pca_plot.jpg")

#----------------------------------------------------------------------------------------------------------------#
   
    def pearson_correlation(self):
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
        prs_corr.to_csv('./result/pearson_correlation.csv')

        print("-------------------- Pearson Correlation --------------------")
        print(tabulate(prs_corr, headers = 'keys', tablefmt = 'psql'),'\n\n')

#----------------------------------------------------------------------------------------------------------------#
  
    def VIF(self):
        #gather features
        temp_df = self.x_train.join(self.y_train)
        features = "+".join(self.x_train.columns.tolist())
        # get y and X dataframes based on this regression:
        y, X = dmatrices(f'{temp_df.columns[-1]} ~' + features, temp_df, return_type='dataframe')
        # For each X, calculate VIF and save in dataframe
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        vif = vif[['features','VIF Factor']].dropna()
        
        print("------ VIF Table -------")
        print(tabulate(vif, headers = 'keys', tablefmt = 'psql'),'\n\n') 

        for i in range(len(vif)):
            if vif['VIF Factor'].iloc[i] > 5:
                print(f'{vif.features.iloc[i]} is highly correlated')
            elif vif['VIF Factor'].iloc[i] == 1:
                print(f'{vif.features.iloc[i]} is not correlated')
            else:
                print(f'{vif.features.iloc[i]} is moderately correlated')

        print('\n\n')
        vif.round(1).to_csv('./result/VIF.csv')
  
#----------------------------------------------------------------------------------------------------------------#
        
    def eigen_vals(self):
        C = np.cov(np.array(self.df), rowvar=False)
        eigvals, _ = np.linalg.eig(C)
        eigen_values = pd.DataFrame(data = eigvals.round(2) , index = self.df.columns , columns = ['eigenvalue'])
        eigen_values = eigen_values[eigen_values['eigenvalue'] > 0]
        eigen_values['condition_number'] = (eigvals.max()/eigen_values.eigenvalue)**(1/2)
        eigen_values = eigen_values[eigen_values['condition_number'] < 30]
        eigen_values.to_csv('./result/eigen_vals.csv')
        print("------ Eigen Values & Condition Number -------")
        print(tabulate(eigen_values, headers = 'keys', tablefmt = 'psql'))
    
#----------------------------------------------------------------------------------------------------------------#
      
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
        
        print("---------- Homogeneity Correlation ----------")
        print(tabulate(homogeneity_corr, headers = 'keys', tablefmt = 'psql'))

#----------------------------------------------------------------------------------------------------------------#