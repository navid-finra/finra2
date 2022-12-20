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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


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

        shaply_df = pd.DataFrame(shap_values[0], columns = self.X_test_scale.columns)
        feature_names = self.X_test_scale.columns

        vals = np.abs(shaply_df.values).mean(0)

        shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                        columns=['col_name','feature_importance_vals'])
        shap_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)

        shap_importance.to_csv('./feat_imp_plots/shap_importance.csv')

    def global_surrogate(self):

        feature_names = self.X_test_scale.columns
        model = load_model(self.model_dirc)
        new_target = model.predict(self.X_test_scale)

        # defining the interpretable decision tree model
        dt_model = DecisionTreeRegressor(max_depth=5, random_state=10)
        dt_model.fit(self.X_train_scale,new_target)
        dt_importances = pd.DataFrame({'feature': feature_names,'importance':np.round(dt_model.feature_importances_,3)})
        dt_importances = dt_importances.sort_values('importance',ascending=False).reset_index(drop = True)
        dt_importances.iloc[:15, :].to_csv('./feat_imp_plots/Global_Surrogate_DT_Importance.csv')  


        forest = RandomForestClassifier(random_state=0)
        forest.fit(self.X_train_scale, new_target)
        forest_importances = pd.DataFrame({'feature': feature_names,'importance':np.round(forest.feature_importances_,3)})
        forest_importances = forest_importances.sort_values('importance',ascending=False).reset_index(drop = True)
        forest_importances.iloc[:15, :].to_csv('./feat_imp_plots/Global_Surrogate_Forest_Importances.csv')
        
        
        xgb_classifier = XGBClassifier()
        xgb_classifier.fit(self.X_train_scale, new_target)
        xgb_importances = pd.DataFrame({'feature': feature_names,'importance':np.round(xgb_classifier.feature_importances_,3)})
        xgb_importances = xgb_importances.sort_values('importance',ascending=False).reset_index(drop = True)
        xgb_importances.iloc[:15, :].to_csv('./feat_imp_plots/Global_Surrogate_XGB_Importances.csv')

    
        ridge_logit = LogisticRegression(C=1, penalty='l2')
        ridge_logit.fit(self.X_train_scale, new_target)
        ridge_importances = pd.DataFrame({'feature': feature_names,'importance':np.round(ridge_logit.coef_[0],3)})
        ridge_importances = ridge_importances.sort_values('importance',ascending=False).reset_index(drop = True)
        ridge_importances.iloc[:15, :].to_csv('./feat_imp_plots/Global_Surrogate_Ridge_Importances.csv')







