from Balance import Balance
from Feature_Redundancy import Feature_Redundancy
from ModelAnalyzer import ModelAnalyzer
from ErrorAnalaysis import ErrorAnalaysis
from FeatureAnalysis import FeatureAnalysis
from Edge_case import Edge_case
from DecisionTreeClass import DecisionTreeClass
from Cluster import Cluster

class Validation(
    Cluster,
    Balance,
    Feature_Redundancy,
    ModelAnalyzer,
    ErrorAnalaysis,
    FeatureAnalysis,
    Edge_case,
    DecisionTreeClass
):
    def __init__(self, data):
        #super(Cluster, self).__init__(data)
        #super(FeatureAnalysis, self).__init__(data)
        #super(ModelAnalyzer, self).__init__(data)
        super(Balance, self).__init__(data)
        super(Edge_case, self).__init__(data)
        super(Feature_Redundancy, self).__init__(data)
        super(ErrorAnalaysis, self).__init__(data)
        super(DecisionTreeClass, self).__init__(data)
