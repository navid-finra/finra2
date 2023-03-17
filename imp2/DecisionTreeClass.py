#----------------------------------------------------------------------------------------------------------------#
from ModelAnalyzer import ModelAnalyzer
#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#----------------------------------------------------------------------------------------------------------------#
import sklearn
from imblearn.over_sampling import SMOTE 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#----------------------------------------------------------------------------------------------------------------#

class DecisionTreeClass:
    def __init__(self,data, analyzer = None):
        
        self.data = data
        if analyzer is None:
            self.model_analysis = ModelAnalyzer(self.data)
        else:
            self.model_analysis = analyzer

        self.model_pred_train = self.model_analysis.model_pred_train
        self.model_pred_test = self.model_analysis.model_pred_test
  
    def decision_tree(self): 
        train_df =  pd.concat([self.data.x_train.reset_index(drop =True), self.data.y_train.reset_index(drop =True)], axis=1)
        model_df_train = pd.DataFrame(self.model_pred, columns = ['pred_label']).reset_index(drop =True)
        train_df = pd.concat([train_df, model_df_train], axis=1)
        test_df =  pd.concat([self.data.x_test.reset_index(drop =True), self.data.y_test.reset_index(drop =True)], axis=1)
        model_df_test = pd.DataFrame(self.model_pred_test, columns = ['pred_label']).reset_index(drop =True)
        test_df = pd.concat([test_df, model_df_test], axis=1)
        new_df = pd.concat([train_df, test_df], axis=0).reset_index(drop =True)
        new_df['fn_error'] = None
        new_df = new_df[new_df['pred_label'] == 0].reset_index(drop =True)

        for r in range(len(new_df)) :
            
            if ((new_df[train_df.columns[-2]][r] == 1)) :
                new_df.fn_error[r] = 1
            else :
                new_df.fn_error[r] = 0
        dt_df = new_df.drop([train_df.columns[-2], 'pred_label'], axis=1)

        x_new = dt_df.drop(['fn_error'], axis=1)
        y_new = dt_df.fn_error.astype('int')
        sm = SMOTE(random_state=42)
        x_blnc, y_blnc = sm.fit_resample(x_new, y_new)
        x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(x_blnc, y_blnc, train_size = 0.7, random_state = 42)
        # List of values to try for max_depth:
        max_depth_range = list(range(1, 15))
        # List to store the accuracy for each value of max_depth:
        test_accuracy = []
        for depth in max_depth_range:

            dt_clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            dt_clf.fit(x_train_dt, y_train_dt)
            score = dt_clf.score(x_test_dt, y_test_dt)
            test_accuracy.append(score)

        fig = plt.figure()
        ax = plt.axes()
        plt.plot(max_depth_range, test_accuracy, color = "yellow")
        plt.xlabel("max depth")
        plt.ylabel("test accuracy")
        plt.title("Tuning the depth of a tree")
        plt.savefig("./result/tuning_tree_plot.jpg")

        dt_clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
        dt_clf.fit(x_train_dt, y_train_dt)

        importances = pd.DataFrame({'feature':x_train_dt.columns,'importance':np.round(dt_clf.feature_importances_,3)})
        self.importances = importances.sort_values('importance',ascending=False)
        self.importances.iloc[:15, :].to_csv('./result/Feature_Importance.csv')  
         
        def tree_to_df(reg_tree, feature_names):
            tree_ = reg_tree.tree_
            feature_name = [
                feature_names[i] if i != sklearn.tree._tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]

            def recurse(node, row, ret):
                if tree_.feature[node] != sklearn.tree._tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    # Add rule to row and search left branch
                    row[-1].append(name + " <= " +  str(round(threshold,3)))
                    recurse(tree_.children_left[node], row, ret)
                    # Add rule to row and search right branch
                    row[-1].append(name + " > " +  str(round(threshold,3)))
                    recurse(tree_.children_right[node], row, ret)
                else:
                    # Add output rules and start a new row
                    label = tree_.value[node]
                    ret.append("Value: [" + str(label[0][0]) + ',' + str(label[0][1]) + "]")
                    row.append([])

            # Initialize
            rules = [[]]
            vals = []

            # Call recursive function with initial values
            recurse(0, rules, vals)

            # Convert to table and output
            df_tree = pd.DataFrame(rules).dropna(how='all')
            df_tree['Return'] = pd.Series(vals)
            columns = []
            for i in range(len(df_tree.columns)-1):
                columns.append(f"Depth {i}")
            columns.append('Values')
            df_tree.columns = columns
            return df_tree

        tree_to_df(dt_clf , self.data.x_train.columns.tolist()).to_csv('./result/decision_tree.csv')    

#----------------------------------------------------------------------------------------------------------------#