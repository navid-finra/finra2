#----------------------------------------------------------------------------------------------------------------#
from finra_classes_main import finra , cluster
from feature_redundancy import feature_redundancy
from balance import balance
from error_analysis import err_cluster
from decision_tree import decision_tree_class
from Analysis import Analysis
from Sensitivity_Analysis import Sensitivity_Analysis_class
from edge_case import edge_case
from continuous_variable import continuous_variable
#----------------------------------------------------------------------------------------------------------------#
import pandas as pd
import argparse
import logging
import pickle
#----------------------------------------------------------------------------------------------------------------#
  
logging.basicConfig(filename="./result/log_file.txt", level=logging.DEBUG)

#----------------------------------------------------------------------------------------------------------------#
  
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_path', type=str, required=True)
parser.add_argument('--train_split_index', type=int, required=True)
parser.add_argument('--labels_column', type=str, required=True)
parser.add_argument('--model_path', type=str)
parser.add_argument('--pickle_in', type=str, required=True)

#----------------------------------------------------------------------------------------------------------------#
  
args = parser.parse_args()
train_split_index = args.train_split_index
labels_column = args.labels_column
model_path = args.model_path
dbfile = open(args.pickle_in, 'rb') 
model = pickle.load(dbfile)
df = pd.read_csv(args.input_data_path, index_col=0).reset_index(drop=True)

#----------------------------------------------------------------------------------------------------------------#

finra_class = finra(df,train_split_index,labels_column,model)
cluster_ = cluster(df,train_split_index,labels_column,model)
feature_redundancy_ = feature_redundancy(df,train_split_index,labels_column,model)
balance_ = balance(df,train_split_index,labels_column,model)
err_cluster_ = err_cluster(df,train_split_index,labels_column,model)
decision_tree_ = decision_tree_class(df,train_split_index,labels_column,model)
Analysis_ = Analysis(df,train_split_index,labels_column,model)
Sensitivity_Analysis_ = Sensitivity_Analysis_class(df,train_split_index,labels_column,model)
continuous_variable_ = continuous_variable(df,train_split_index,labels_column,model)
edge_case_ = edge_case(df,train_split_index,labels_column,model)

#----------------------------------------------------------------------------------------------------------------#

cluster_.number_of_cluster()

feature_redundancy_.pca()

feature_redundancy_.pireson_correlation()

feature_redundancy_.VIF()

feature_redundancy_.eigen_vals()

feature_redundancy_.homogeneity_corr()

balance_.Balance_check()

err_cluster_.error_cluster()

err_cluster_.train_score()

err_cluster_.test_score()

err_cluster_.false_negative()

decision_tree_.decision_tree()

Analysis_.PDP()

Analysis_.ICE()

Analysis_.ALE()

Analysis_.Permutation_feature_importance()

Analysis_.Global_Surrogate()

Analysis_.LIME()

Sensitivity_Analysis_.Sensitivity_Analysis()

continuous_variable_.r2()

continuous_variable_.mse()

continuous_variable_.chi_square()

continuous_variable_.Kolmogorov_Smirnov()

edge_case_.edge_case_analysis()

#----------------------------------------------------------------------------------------------------------------#