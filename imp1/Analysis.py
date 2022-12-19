import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor

import shap
from PyALE import ale
import eli5
from eli5.sklearn import PermutationImportance

plt.style.use('fivethirtyeight')

import os
try:   
    os.mkdir('./feat_imp_plots')
except:
    pass

class Analysis:
    def __init__(self, X_train, X_test, y_train , y_test,  model_dirc):

        self.X_train_scale = X_train
        self.X_test_scale = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_dirc =  model_dirc
        def base_model():
            model = load_model(self.model_dirc)
            return model
        self.my_model = KerasRegressor(build_fn=base_model)    
        self.my_model.fit(self.X_train_scale,self.y_train)

    def feature_importance_values(self):    
        perm = PermutationImportance(self.my_model, random_state=1).fit(self.X_test_scale,self.y_test)
        feat_imp = pd.DataFrame(perm.feature_importances_, index = self.X_test_scale.columns, columns = ['importance']).sort_values(by =['importance'], ascending = False)
        feat_imp.to_csv('./feat_imp_plots/feat_imp_df.csv')

    def PDP(self):
        try:   
            os.mkdir('./feat_imp_plots/PDP')
        except:
            pass
        plt.figure(figsize=(100, 300), dpi=100)
        for f in self.X_test_scale.columns :  
            shap.plots.partial_dependence(
                f, self.my_model.predict, self.X_test_scale, ice=False, pd_linewidth = 3, show = False)
            plt.title(f'{f}')
            plt.tight_layout()
            plt.savefig(f'./feat_imp_plots/PDP/PDP_{f}.png')
            plt.clf()

    def ICE(self):
        try:   
            os.mkdir('./feat_imp_plots/ICE')
        except:
            pass
        for f in self.X_test_scale.columns :  
            shap.plots.partial_dependence(
                f, self.my_model.predict, self.X_test_scale, ice=True, pd_linewidth = 3, show = False
                )

            plt.title(f'{f}')
            plt.tight_layout()
            plt.savefig(f'./feat_imp_plots/ICE/ICE_{f}.png')
            plt.clf()

    def ALE(self):
        try:   
            os.mkdir('./feat_imp_plots/ALE')
        except:
            pass
        model = load_model(self.model_dirc)
        for f in self.X_test_scale.columns :  
            ale(X = self.X_test_scale, model=model, feature=[f], grid_size=50, include_CI=True, C = 0.95)
            
            plt.title(f'{f}')
            plt.tight_layout()
            plt.savefig(f'./feat_imp_plots/ALE/ALE_{f}.png')
            plt.clf()

        try:   
            os.mkdir('./feat_imp_plots/ALE_2D')
        except:
            pass

        perm = PermutationImportance(self.my_model, random_state=1).fit(self.X_test_scale,self.y_test)
        feat_imp = pd.DataFrame(perm.feature_importances_, index = self.X_test_scale.columns, columns = ['importance']).sort_values(by =['importance'], ascending = False)

        feature_number = 5
        feats = feat_imp.index[:feature_number]
        for f1 in range(feature_number) :  
            for f2 in range(f1+1, feature_number) :  
                ale(X=self.X_test_scale, model=model, feature=[feats[f1],feats[f2]], grid_size=100)
                plt.title(f'{feats[f1]}&{feats[f2]}')
                plt.tight_layout()
                plt.savefig(f'./feat_imp_plots/ALE_2D/ALE_2D_{feats[f1]}_{feats[f2]}.png')
                plt.clf()
    
    def shaply(self):
        model = load_model(self.model_dirc)

        e = shap.KernelExplainer(model, self.X_train_scale)
        shap_values = e.shap_values(self.X_test_scale)

        shaply_df = pd.DataFrame(shap_values[0], columns = self.X_train_scale.columns)
        feature_names = self.X_train_scale.columns

        vals = np.abs(shaply_df.values).mean(0)

        shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                        columns=['col_name','feature_importance_vals'])
        shap_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)

        shap_importance.to_csv('./feat_imp_plots/shaply_importance.csv')

    
