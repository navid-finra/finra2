#----------------------------------------------------------------------------------------------------------------#
from decision_tree import decision_tree
#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#----------------------------------------------------------------------------------------------------------------#
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.inspection import PartialDependenceDisplay 
from sklearn.inspection import permutation_importance
from PyALE import ale
from lime.lime_tabular import LimeTabularExplainer
#----------------------------------------------------------------------------------------------------------------#

class Analysis(decision_tree):

    def __init__(self, df, train_split_index, labels_column, model):
        super().__init__(df, train_split_index, labels_column, model)

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
        self.x_train_dt, self.x_test_dt, self.y_train_dt, self.y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(self.x_train_dt, self.y_train_dt)
            score = dt_clf.score(self.x_test_dt, self.y_test_dt)
            test_accuracy.append(score)

        self.dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        self.dt_clf.fit(self.x_train_dt, self.y_train_dt)

#----------------------------------------------------------------------------------------------------------------#

    def PDP(self):
        importances = pd.DataFrame({'feature':self.x_train_dt.columns,'importance':np.round(self.dt_clf.feature_importances_,3)})
        importances = importances.sort_values('importance',ascending=False)
        features = importances.feature[:2].tolist()
        PartialDependenceDisplay.from_estimator(self.dt_clf,
                                       X = self.x_train_dt,
                                       features = features, 
                                       target=0)
        plt.savefig("./result/PDP_plot.jpg")
    
#----------------------------------------------------------------------------------------------------------------#

    def ICE(self):
        importances = pd.DataFrame({'feature':self.x_train_dt.columns,'importance':np.round(self.dt_clf.feature_importances_,3)})
        importances = importances.sort_values('importance',ascending=False)
        features = importances.feature[:2].tolist()
        PartialDependenceDisplay.from_estimator(self.dt_clf, self.x_train_dt, features ,kind='individual')
        plt.savefig("./result/ICE_plot.jpg")

#----------------------------------------------------------------------------------------------------------------#    

    def ALE(self):
        importances = pd.DataFrame({'feature':self.x_train_dt.columns,'importance':np.round(self.dt_clf.feature_importances_,3)})
        importances = importances.sort_values('importance',ascending=False)
        features = importances.feature[:2].tolist()
        ale_eff = ale(X=self.x_train_dt, model=self.dt_clf, feature=[features[0]], grid_size=50, include_CI=True, C=0.95)
        plt.savefig("./result/ALE_1D_plot.jpg")
        ale_eff2D = ale(X=self.x_train_dt, model=self.dt_clf, feature=features, grid_size=100)
        plt.savefig("./result/ALE_2D_plot.jpg")

#----------------------------------------------------------------------------------------------------------------#

    def Permutation_feature_importance(self):
        r = permutation_importance(self.dt_clf, self.x_train_dt, self.y_train_dt,
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

#----------------------------------------------------------------------------------------------------------------#

    def Global_Surrogate(self):
        new_target = self.dt_clf.predict(self.x_train_dt)

        # defining the interpretable decision tree model
        dt_model1 = DecisionTreeRegressor(max_depth=5, random_state=10)

        # fitting the surrogate decision tree model using the training set and new target
        dt_model1.fit(self.x_train_dt,new_target)

#----------------------------------------------------------------------------------------------------------------#

    def LIME(self):
        explainer = LimeTabularExplainer(self.x_train_dt.values, mode="regression", feature_names=self.x_train_dt.columns)
        r = permutation_importance(self.dt_clf, self.x_train_dt, self.y_train_dt,
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
            X_observation = self.x_test_dt.iloc[[i], :]
            feature_name.append(self.x_train.columns[i])
            feature_value.append(self.dt_clf.predict(X_observation)[0])
           
        RF_prediction_df = pd.DataFrame({'feature':feature_name , 'value':feature_value})
        RF_prediction_df.to_csv('./result/RF_prediction.csv')

        explanation = explainer.explain_instance(X_observation.values[0], self.dt_clf.predict)
        explanation.show_in_notebook(show_table=True, show_all=False)

#----------------------------------------------------------------------------------------------------------------#