import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite, data_integrity
from deepchecks.tabular.checks import LabelDrift

class Deep_Check:
    def __init__(self, data):
        self.data = data

    def deep_check(self, target):
        model = self.data.model
        x_train = self.data.x_train
        x_test = self.data.x_test
        y_train = self.data.y_train
        y_test = self.data.y_test
        X_train = pd.concat([x_train, y_train], axis=1)
        X_test = pd.concat([x_test, y_test], axis=1)
        ds_train = Dataset(X_train, label=target, cat_features=[])
        ds_test =  Dataset(X_test,  label=target, cat_features=[])
        suite = full_suite()
        fullsuite = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model)
        integ_suite = data_integrity()
        integrity_suite = integ_suite.run(ds_train)
        check = LabelDrift()
        result = check.run(ds_train, ds_test)
        return fullsuite, integrity_suite, result