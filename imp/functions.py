#----------------------------------------------------------------------------------------------------------------#
from finra_classes_main import finra , cluster
from feature_redundancy import feature_redundancy
from balance import balance
from error_analysis import err_cluster
from decision_tree import decision_tree_class
from Analysis import Analysis
from Sensitivity_Analysis import Sensitivity_Analysis_class
#----------------------------------------------------------------------------------------------------------------#

finra_class = finra(df,train_split_index,labels_column,model)
cluster_ = cluster(df,train_split_index,labels_column,model)
feature_redundancy_ = feature_redundancy(df,train_split_index,labels_column,model)
balance_ = balance(df,train_split_index,labels_column,model)
err_cluster_ = err_cluster(df,train_split_index,labels_column,model)
decision_tree_ = decision_tree_class(df,train_split_index,labels_column,model)
Analysis_ = Analysis(df,train_split_index,labels_column,model)
Sensitivity_Analysis_ = Sensitivity_Analysis_class(df,train_split_index,labels_column,model)

#----------------------------------------------------------------------------------------------------------------#

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

#----------------------------------------------------------------------------------------------------------------#