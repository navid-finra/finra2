import pandas as pd

class Data:
    def __init__(
        self, 
        x_train, 
        y_train, 
        x_test, 
        y_test,
        model, 
        analysis_type = 'regular', 
        model_type = 'NeuralNet'
        ):

        try:
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            self.model = model
            self.optimum = None
            self.df = pd.concat([self.x_train, self.x_test])
            self.cluster_group = None
            self.analysis_type = analysis_type
            self.model_type = model_type
            print('Data object created successfully')
        except:
            print('Data object not created please check the inputs')

    def show(self):
        print('\nTraining Data:\n')
        print(self.x_train.head())
        print('\nTest Data:\n')
        print(self.x_test.head())
        print('\nTraining Labels\n')
        print(self.y_train.head())
        print('\nTest Labels\n')
        print(self.y_test.head())
        print('\nNumber of Clusters:\n')
        print(self.optimum)
        print('\nCluster Groups:\n')
        print(self.cluster_group)

    def get_info(self, data_type='train'):

        if data_type.lower() == 'train':
            df = self.x_train
            label = self.y_train
        else:
            df = self.x_test
            label = self.y_test

        ind, val = [], []
        ind.append('Features')
        val.append(df.shape[1])
        try:
            val.append(label.shape[1])
            ind.append('Targets')    
        except: pass

        ind.append('Samples')
        val.append(len(df))
        ind.append('Dimension')
        val.append(str(df.shape))
        ind.append('Duplicated')
        val.append(df.duplicated().sum())
        ind.append('Missing Values')
        val.append(df.isnull().sum().sum())
        ind.append('Negative Samples')
        df_join = df.join(label)
        val.append(df_join.groupby(df_join.columns[-1]).size()[0])
        ind.append('Positive Samples')
        val.append(df_join.groupby(df_join.columns[-1]).size()[1])
        ind.append('Balance (Negative / Positive)')
        val.append(df_join.groupby(df_join.columns[-1]).size()[0] / df_join.groupby(df_join.columns[-1]).size()[1])
        rep_df = pd.DataFrame({'ind':ind, 'val':val})
        return rep_df