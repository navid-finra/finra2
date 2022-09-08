from finra_classes import finra
import pandas as pd
import argparse
import os
import logging
import pickle

logging.basicConfig(filename="./result/log_file.log", encoding='utf-8', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--input_data_path', type=str, required=True)
parser.add_argument('--train_split_index', type=int, required=True)
parser.add_argument('--labels_column', type=str, required=True)
parser.add_argument('--model_path', type=str)
parser.add_argument('--pickle_in', type=str, required=True)
parser.add_argument('--pca', type=str)
parser.add_argument('--pireson_correlation', type=str)
parser.add_argument('--VIF', type=str)
parser.add_argument('--eigen_vals', type=str)
parser.add_argument('--homogeneity_corr', type=str)
parser.add_argument('--Balance_check', type=str)
parser.add_argument('--Error_analysis_train', type=str)
parser.add_argument('--Error_analysis_test', type=str)
parser.add_argument('--confusion_matrix_train', type=str)
parser.add_argument('--confusion_matrix_test', type=str)
parser.add_argument('--false_negative', type=str)
parser.add_argument('--decision_tree', type=str)
parser.add_argument('--PDP', type=str)
parser.add_argument('--ICE', type=str)
parser.add_argument('--ALE', type=str)
parser.add_argument('--Permutation_feature_importance', type=str)
parser.add_argument('--Global_Surrogate', type=str)
parser.add_argument('--LIME', type=str)
parser.add_argument('--edge_case', type=str)

args = parser.parse_args()
train_split_index = args.train_split_index
labels_column = args.labels_column
model_path = args.model_path
dbfile = open(args.pickle_in, 'rb') 
model = pickle.load(dbfile)
logging.debug(f'model namet = {model}')
logging.debug('-------------------------------------------------------------------------------')

df = pd.read_csv(args.input_data_path, index_col=0).reset_index(drop=True)
finra_class = finra(df,train_split_index,labels_column,model)

if args.pca:
    finra_class.pca()
    logging.debug('PCA was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.pireson_correlation:
    finra_class.pireson_correlation()
    logging.debug('pireson_correlation was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.VIF:
    finra_class.VIF()
    logging.debug('VIF was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.eigen_vals:
    finra_class.eigen_vals()
    logging.debug('eigen_vals was done!')
    logging.debug('-------------------------------------------------------------------------------')
    
if args.homogeneity_corr:
    finra_class.homogeneity_corr()
    logging.debug('homogeneity_corr was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.Balance_check:
    finra_class.Balance_check()
    logging.debug('Balance_check was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.Error_analysis_train:
    finra_class.Error_analysis_train()
    logging.debug('Error_analysis_train was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.Error_analysis_test:
    finra_class.Error_analysis_test()
    logging.debug('Error_analysis_test was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.confusion_matrix_train:
    finra_class.confusion_matrix_train()
    logging.debug('confusion_matrix_train was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.confusion_matrix_test:
    finra_class.confusion_matrix_test()
    logging.debug('confusion_matrix_test was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.false_negative:
    finra_class.false_negative()
    logging.debug('false_negative was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.decision_tree:
    finra_class.decision_tree()
    logging.debug('decision_tree was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.PDP:
    finra_class.PDP()
    logging.debug('PDP was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.ICE:
    finra_class.ICE()
    logging.debug('ICE was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.ALE:
    finra_class.ALE()
    logging.debug('ALE was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.Permutation_feature_importance:
    finra_class.Permutation_feature_importance()
    logging.debug('Permutation_feature_importance was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.Global_Surrogate:
    finra_class.Global_Surrogate()
    logging.debug('Global_Surrogate was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.LIME:
    finra_class.LIME()
    logging.debug('LIME was done!')
    logging.debug('-------------------------------------------------------------------------------')

if args.edge_case:
    finra_class.edge_case_analysis()
    logging.debug('edge_case_analysis was done!')
    logging.debug('-------------------------------------------------------------------------------')

### command for run 
#python finra.py --input_data_path  "./data/input_data.csv" --train_split_index 3980 --labels_column "target" --pickle_in "examplePickle" --pca True --edge_case True