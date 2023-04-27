import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil

from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from keras.wrappers.scikit_learn import KerasRegressor
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer
from eli5.sklearn import PermutationImportance
from PyALE import ale
import shap
shap.initjs()




import os
try:   
    os.mkdir('./feat_imp_plots')
except:
    pass

class FeatureAnalysis:
    def __init__(self, data, analyzer = None):

        self.data = data
        def base_model():
            if self.data.analysis_type.lower() == 'binary':
                model = tf.keras.models.clone_model(self.data.model)
                model.add(tf.keras.layers.Lambda(lambda x:x[:,1]))
                model.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics = [tf.keras.metrics.AUC(name='auc')])
            else:
                model = self.data.model
            return model
        if self.data.model_type =='NeuralNet':
            self.my_model = KerasRegressor(build_fn=base_model)
            self.my_model.fit(self.data.x_train, self.data.y_train)
        else:
            self.my_model = self.data.model
            self.my_model.fit(self.data.x_train, self.data.y_train)

        self.cols = []
        for col in self.data.x_train.columns:
            if len(self.data.x_train[col].unique()) >= 2:
                self.cols.append(col)


    def feature_importance_values(self):    
        perm = PermutationImportance(self.my_model, random_state=1).fit(self.data.x_test,self.data.y_test)
        feat_imp = pd.DataFrame(perm.feature_importances_, index = self.data.x_test.columns, columns = ['importance']).sort_values(by =['importance'], ascending = False)
        feat_imp.to_csv('./feat_imp_plots/feat_imp_df.csv')
        return feat_imp.reindex(feat_imp.importance.abs().sort_values().index)

    def pdp(self):
        try:   
            os.mkdir('./feat_imp_plots/PDP')
        except:
            pass
        plt.figure(figsize=(100, 300), dpi=100)
        for f in self.data.x_test.columns :  
            shap.plots.partial_dependence(
                f, self.my_model.predict, self.data.x_test, ice=False, pd_linewidth = 3, show = False)
            plt.title(f'{f}')
            plt.tight_layout()
            plt.savefig(f'./feat_imp_plots/PDP/PDP_{f}.png')
            plt.clf()
        shutil.make_archive('PDP', 'zip', './feat_imp_plots/PDP')

    def ice(self):
        try:   
            os.mkdir('./feat_imp_plots/ICE')
        except:
            pass
        for f in self.cols :  
            shap.plots.partial_dependence(
                f, self.my_model.predict, self.data.x_test, ice=True, pd_linewidth = 3, show = False
                )

            plt.title(f'{f}')
            plt.tight_layout()
            plt.savefig(f'./feat_imp_plots/ICE/ICE_{f}.png')
            plt.clf()
        shutil.make_archive('ICE', 'zip', './feat_imp_plots/ICE')

    def ALE(self, two_d = False):
        try:   
            os.mkdir('./feat_imp_plots/ALE')
        except:
            pass
        for f in self.data.x_test.columns :  
            try:
                ale(X = self.data.x_test, model=self.my_model, feature=[f], grid_size=50, include_CI=True, C = 0.95)
                plt.title(f'{f}')
                plt.tight_layout()
                plt.savefig(f'./feat_imp_plots/ALE/ALE_{f}.png')
                plt.clf()
            except:
                pass
        shutil.make_archive('ALE1D', 'zip', './feat_imp_plots/ALE')

        perm = PermutationImportance(self.my_model, random_state=1).fit(self.data.x_test,self.data.y_test)
        feat_imp = pd.DataFrame(perm.feature_importances_, index = self.data.x_test.columns, columns = ['importance']).sort_values(by =['importance'], ascending = False)

        if two_d:
            try:   
                os.mkdir('./feat_imp_plots/ALE_2D')
            except:
                pass
            
            feature_number = 5
            feats = feat_imp.index[:feature_number]
            for f1 in range(feature_number) :  
                for f2 in range(f1+1, feature_number) :  
                    try:
                        ale(X=self.data.x_test, model=self.my_model, feature=[feats[f1],feats[f2]], grid_size=100)
                        plt.title(f'{feats[f1]}&{feats[f2]}')
                        plt.tight_layout()
                        plt.savefig(f'./feat_imp_plots/ALE_2D/ALE_2D_{feats[f1]}_{feats[f2]}.png')
                        plt.clf()
                    except:
                        pass
            shutil.make_archive('ALE2D', 'zip', './feat_imp_plots/ALE_2D')

    def shaply(self):

        if self.data.model_type == 'NeuralNet':
            explainer = shap.KernelExplainer(self.my_model, self.data.x_train)
            shap_values = explainer.shap_values(self.data.x_test)
            

        else: 
            explainer = shap.TreeExplainer(self.my_model)
            shap_values = explainer.shap_values(self.data.x_train)
        try:
            shaply_df = pd.DataFrame(shap_values[0], columns = self.data.x_test.columns)
        except:
            shaply_df = pd.DataFrame(shap_values, columns = self.data.x_test.columns)
        feature_names = self.data.x_test.columns
        vals = np.abs(shaply_df.values).mean(0)
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                        columns=['col_name','feature_importance_vals'])
        shap_importance.sort_values(by=['feature_importance_vals'],
                            ascending=False, inplace=True)
        shap_importance = shap_importance.reset_index(drop = True)
        shap_importance.to_csv('./feat_imp_plots/shap_importance.csv')


    def LIME(self, sample_list):

        try:   
            os.mkdir('./feat_imp_plots/ALE')
        except:
            pass

        if self.data.model_type == 'NeuralNet':
            print('Lime analysis is not possible')

        else:
            for x_sample in sample_list:
                explainer = LimeTabularExplainer(self.data.x_train.values, mode="regression", feature_names=self.data.x_train.columns)
                explanation = explainer.explain_instance(self.data.x_train.loc[x_sample,self.data.x_train.columns], self.my_model.predict)
                explanation.as_pyplot_figure()
                plt.savefig(f'./result/feat_imp_plots/LIME{x_sample}.png')
                explanation_list = [[i,round(j,3)] for i, j in explanation.as_list()]
                for i in explanation_list:
                    print(i)


    def cv_kfolds(self, k = 10):

        n_splits = k
        kf = KFold(n_splits = n_splits, random_state=1, shuffle=True)
        scores = np.round(
            cross_val_score(
                self.my_model, 
                np.array(self.data.x_train), 
                np.array(self.data.y_train), 
                cv=kf, 
                n_jobs=-1
            ),2
        )
        print(f'k-fold cross validation score for k={n_splits} :', list(scores))
        print(f'k-fold cross validation score for k={n_splits} has an standard deviation of {round(statistics.stdev(list(scores)), 3)} and mean of {statistics.mean(list(scores))}')


    def global_surrogate(self):

        feature_names = self.data.x_test.columns
        if self.data.analysis_type == 'binary':
            new_target = np.argmax(self.data.model.predict(self.data.x_train), axis=1)
        else:
            new_target = self.data.y_train

        # defining the interpretable decision tree model
        dt_model = DecisionTreeRegressor(max_depth=5, random_state=10)
        dt_model.fit(self.data.x_train,new_target)
        dt_importances = pd.DataFrame({'feature': feature_names,'importance':np.round(dt_model.feature_importances_,3)})
        dt_importances = dt_importances.sort_values('importance',ascending=False).reset_index(drop = True)
        dt_importances.iloc[:15, :].to_csv('./feat_imp_plots/Global_Surrogate_DT_Importance.csv')  

        forest = RandomForestClassifier(random_state=0)
        forest.fit(self.data.x_train, new_target)
        forest_importances = pd.DataFrame({'feature': feature_names,'importance':np.round(forest.feature_importances_,3)})
        forest_importances = forest_importances.sort_values('importance',ascending=False).reset_index(drop = True)
        forest_importances.iloc[:15, :].to_csv('./feat_imp_plots/Global_Surrogate_Forest_Importances.csv')
        
        xgb_classifier = XGBClassifier()
        xgb_classifier.fit(self.data.x_train, new_target)
        xgb_importances = pd.DataFrame({'feature': feature_names,'importance':np.round(xgb_classifier.feature_importances_,3)})
        xgb_importances = xgb_importances.sort_values('importance',ascending=False).reset_index(drop = True)
        xgb_importances.iloc[:15, :].to_csv('./feat_imp_plots/Global_Surrogate_XGB_Importances.csv')
    
        ridge_logit = LogisticRegression(C=1, penalty='l2')
        ridge_logit.fit(self.data.x_train, new_target)
        ridge_importances = pd.DataFrame({'feature': feature_names,'importance':np.round(ridge_logit.coef_[0],3)})
        ridge_importances = ridge_importances.sort_values('importance',ascending=False).reset_index(drop = True)
        ridge_importances.iloc[:15, :].to_csv('./feat_imp_plots/Global_Surrogate_Ridge_Importances.csv')