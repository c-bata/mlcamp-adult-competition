import pandas as pd
from sklearn import preprocessing
from typing import Tuple

from automl.rfe import RFE


def generate_feature(train_x: pd.DataFrame, train_y: pd.DataFrame,
                     test_x: pd.DataFrame, n_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_feature = train_x.copy()
    test_feature = test_x.copy()

    # preprocess
    train_feature = train_feature.fillna(value='?')
    test_feature = test_feature.fillna(value='?')

    # categorical encoding
    train_feature['sex'] = train_feature['sex'] == 'Male'
    test_feature['sex'] = test_feature['sex'] == 'Male'

    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship',
        'race', 'capital-gain', 'native-country']
    features = train_feature.append(test_feature, ignore_index=True)
    for column in categorical_columns:
        le = preprocessing.LabelEncoder()
        le.fit(features[column])
        train_feature[column] = pd.Series(le.transform(train_feature[column])).astype('category')
        test_feature[column] = pd.Series(le.transform(test_feature[column])).astype('category')

    # feature selection
    selector = RFE(n_features_to_select=n_features)
    selector.fit(train_feature, train_y)

    train_feature = train_feature[train_feature.columns[selector.support_]]
    test_feature = test_feature[test_feature.columns[selector.support_]]
    return train_feature, test_feature
