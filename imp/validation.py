#----------------------------------------------------------------------------------------------------------------#
from validation_classes_main import validation, cluster
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
parser.add_argument('--pca', type=str)
parser.add_argument('--pearson_correlation', type=str)
parser.add_argument('--VIF', type=str)
parser.add_argument('--eigen_vals', type=str)
parser.add_argument('--homogeneity_corr', type=str)
parser.add_argument('--Balance_check', type=str)
parser.add_argument('--error_cluster', type=str)
parser.add_argument('--train_score', type=str)
parser.add_argument('--test_score', type=str)
parser.add_argument('--false_negative', type=str)
parser.add_argument('--decision_tree', type=str)
parser.add_argument('--PDP', type=str)
parser.add_argument('--ICE', type=str)
parser.add_argument('--ALE', type=str)
parser.add_argument('--Permutation_feature_importance', type=str)
parser.add_argument('--Global_Surrogate', type=str)
parser.add_argument('--LIME', type=str)
parser.add_argument('--Sensitivity_Analysis', type=str)
parser.add_argument('--edge_case_analysis', type=str)
parser.add_argument('--r2', type=str)
parser.add_argument('--mse', type=str)
parser.add_argument('--chi_square', type=str)
parser.add_argument('--Kolmogorov_Smirnov', type=str)
#----------------------------------------------------------------------------------------------------------------#
  
args = parser.parse_args()
train_split_index = args.train_split_index
labels_column = args.labels_column
model_path = args.model_path
dbfile = open(args.pickle_in, 'rb') 
model = pickle.load(dbfile)
df = pd.read_csv(args.input_data_path, index_col=0).reset_index(drop=True)

#----------------------------------------------------------------------------------------------------------------#
  
validation_class = validation(df,train_split_index,labels_column,model)
cluster_ = cluster(df,train_split_index,labels_column,model)
feature_redundancy_ = feature_redundancy(df,train_split_index,labels_column,model)
balance_ = balance(df,train_split_index,labels_column,model)
err_cluster_ = err_cluster(df,train_split_index,labels_column,model)
decision_tree_ = decision_tree_class(df,train_split_index,labels_column,model)
Analysis_ = Analysis(df,train_split_index,labels_column,model)
Sensitivity_Analysis_ = Sensitivity_Analysis_class(df,train_split_index,labels_column,model)
Edge_case_ = edge_case(df,train_split_index,labels_column,model)
continuous_variable_ = continuous_variable(df,train_split_index,labels_column,model)

#----------------------------------------------------------------------------------------------------------------#

if args.pca:
    feature_redundancy_.pca()

if args.pearson_correlation:
    feature_redundancy_.pearson_correlation()

if args.VIF:
    feature_redundancy_.VIF()

if args.eigen_vals:
    feature_redundancy_.eigen_vals()

if args.homogeneity_corr:
    feature_redundancy_.homogeneity_corr()

if args.Balance_check:
    balance_.Balance_check()

if args.error_cluster:
    err_cluster_.error_cluster()

if args.train_score:
    err_cluster_.train_score()

if args.test_score:
    err_cluster_.test_score()

if args.false_negative:
    err_cluster_.false_negative()

if args.decision_tree:
    decision_tree_.decision_tree()

if args.PDP:
    Analysis_.PDP()

if args.ICE:
    Analysis_.ICE()

if args.ALE:
    Analysis_.ALE()

if args.Permutation_feature_importance:
    Analysis_.Permutation_feature_importance()

if args.Global_Surrogate:
    Analysis_.Global_Surrogate()

if args.LIME:
    Analysis_.LIME()

if args.LIME:
    Analysis_.shaply()

if args.Sensitivity_Analysis:
    Sensitivity_Analysis_.Sensitivity_Analysis()

if args.edge_case_analysis:
    Edge_case_.edge_case_analysis()

if args.r2:
    continuous_variable_.r2()

if args.mse:
    continuous_variable_.mse()

if args.chi_square:
    continuous_variable_.chi_square()

if args.Kolmogorov_Smirnov:
    continuous_variable_.Kolmogorov_Smirnov()

#----------------------------------------------------------------------------------------------------------------#

### command for run 
#python validation.py --input_data_path  "./data/input_data.csv" --train_split_index 1000 --labels_column "target" --pickle_in "examplePickle" --pca True