import numpy as np
import pandas as pd
from scipy import stats
from sklearn import tree
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV


class Data:

    def __init__(self):
        # bd
        self.bank_data = self._get_bank_data()
        self.data_with_dummies = self._get_data_with_dummies(bank_data=self.bank_data)

        self.upsampled_data_with_outliers = self._upsample(data=self.data_with_dummies)
        self.downsampled_data_with_outliers = self._downsample(data=self.data_with_dummies)

        self.upsampled_x_with_outliers = self._gridsearch_classification(data=self.upsampled_data_with_outliers)
        self.downsampled_x_with_outliers = self._gridsearch_classification(data=self.downsampled_data_with_outliers)

        self.upsampled_data_without_outliers = self._upsample(data=self._remove_outliers(data=self.bank_data))
        self.downsampled_data_without_outliers = self._downsample(data=self._remove_outliers(data=self.bank_data))

        self.upsampled_x_without_outliers = self._gridsearch_classification(data=self.upsampled_data_without_outliers)
        self.downsampled_x_without_outliers = self._gridsearch_classification(data=self.downsampled_data_without_outliers)

    def _get_bank_data(self):
        bank = pd.read_csv('bank-full.csv', sep=';')

        bank_data = bank.copy()
        bank_data['poutcome'] = bank_data['poutcome'].replace(['unknown'] , 'other')
        bank_data.poutcome.value_counts()
        bank_data.drop('contact', axis=1, inplace=True)
        bank_data['credit_default'] = bank_data['default'].map( {'yes':1, 'no':0} )
        bank_data.drop('default', axis=1,inplace = True)
        bank_data["housing_loan"]=bank_data['housing'].map({'yes':1, 'no':0})
        bank_data.drop('housing', axis=1,inplace = True)
        bank_data["personal_loan"] = bank_data['loan'].map({'yes':1, 'no':0})
        bank_data.drop('loan', axis=1, inplace=True)
        bank_data.drop('month', axis=1, inplace=True)
        bank_data.drop('day', axis=1, inplace=True)
        bank_data.drop('duration', axis=1, inplace=True)
        bank_data["deposit"] = bank_data['y'].map({'yes':1, 'no':0})
        bank_data.drop('y', axis=1, inplace=True)

        return bank_data

    def _get_data_with_dummies(self, bank_data):
        bd = bank_data
        bd['job'] = bd['job'].replace(['unknown'] , 'other')
        bd['job'] = bd['job'].replace(['student'] , 'other')
        bd['job'] = bd['job'].replace(['housemaid'] , 'other')
        bd['job'] = bd['job'].replace(['unemployed'] , 'other')
        bd['job'] = bd['job'].replace(['entrepreneur'] , 'other')
        bd['job'] = bd['job'].replace(['self-employed'] , 'other')
        bd['job'] = bd['job'].replace(['retired'] , 'other')
        bd['job'] = bd['job'].replace(['services'] , 'other')

        bank_with_dummies = pd.get_dummies(data=bank_data, columns = ['job', 'marital', 'education', 'poutcome'], prefix = ['job', 'marital', 'education', 'poutcome'])

        return bank_with_dummies

    # data is heavily imbalance, which cause bias towards 'no'
    def _upsample(self, data):
        # data is heavily imbalanced and the decision tree is biased to always return yes, resampling to balance classes

        # Separate majority and minority classes
        df_majority = data[data.deposit==0]
        df_minority = data[data.deposit==1]

        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=df_majority.shape[0],
                                         random_state=123)

        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])

        return df_upsampled

    def _downsample(self, data):
        # Separate majority and minority classes
        df_majority = data[data.deposit==0]
        df_minority = data[data.deposit==1]

        # Downsample majority class
        df_majority_downsampled = resample(df_majority,
                                           replace=False,    # sample without replacement
                                           n_samples=df_minority.shape[0],     # to match minority class
                                           random_state=123) # reproducible results

        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        return df_downsampled

    def _remove_outliers(self, data):
            bd = data
            bd_dummies = pd.get_dummies(data=bd, columns = ['job', 'marital', 'education', 'poutcome'], prefix = ['job', 'marital', 'education', 'poutcome'])
            data_outliers = bd_dummies
            without = data_outliers[(np.abs(stats.zscore(data_outliers)) < 3).all(axis=1)]
            return without

    def _gridsearch_classification(self, data, scoring=None):
        y = data.deposit
        x = data.drop('deposit',1)
        parameters = {'max_depth':range(3,6), 'criterion':['gini', 'entropy'], 'splitter':['random', 'best']}
        clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4, return_train_score=True)
        clf.fit(X=x, y=y)
        tree_model = clf.best_estimator_
        # print ('score and params: ', clf.best_score_, clf.best_params_)
        # print('train: ', np.mean(clf.cv_results_['mean_train_score']))
        # prob_y = clf.predict_proba(x)
        # prob_y = [p[1] for p in prob_y]
        # print('roc: ', roc_auc_score(y, prob_y) )
        return tree_model
