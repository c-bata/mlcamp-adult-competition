import featuretools as ft
import pandas as pd
from typing import Tuple, Optional
from sklearn import preprocessing

from automl.rfe import RFE


def generate_feature(train_x: pd.DataFrame,
                     train_y: pd.DataFrame,
                     test_x: pd.DataFrame,
                     n_features: Optional[int],
                     synthesis: bool = True,
                     category_encoding: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_feature = train_x.copy()
    test_feature = test_x.copy()

    # preprocess (impute missing values for label encoding)
    train_feature = train_feature.fillna(value='?')
    test_feature = test_feature.fillna(value='?')

    train_feature['sex'] = train_feature['sex'] == 'Male'
    test_feature['sex'] = test_feature['sex'] == 'Male'

    # target encoding
    target_encoding_columns = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship',
        'race', 'capital-gain', 'native-country']

    for column in target_encoding_columns:
        train_feature[f'{column}_target_encoding'] = 0
        for value in train_feature[column].unique():
            value_mean = train_y[train_feature[column] == value].mean()
            train_feature.loc[train_feature[column] == value, f'{column}_target_encoding'] = value_mean
            test_feature.loc[test_feature[column] == value, f'{column}_target_encoding'] = value_mean

    # frequency encoding
    frequency_encoding_columns = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship',
        'race', 'capital-gain', 'native-country']

    for column in frequency_encoding_columns:
        freq = train_feature[column].value_counts()
        train_feature[f'{column}_frequency_encoding'] = train_feature[column].map(freq)
        test_feature[f'{column}_frequency_encoding'] = test_feature[column].map(freq)

    if category_encoding == 'ohe':
        # one-hot encoding
        features = train_feature.append(test_feature, ignore_index=True)
        len_train_feature = len(train_feature)
        updated = pd.get_dummies(features)
        train_feature = updated[:len_train_feature]
        test_feature = updated[len_train_feature:]
    elif category_encoding == 'label':
        # label encoding
        features = train_feature.append(test_feature, ignore_index=True)
        categorical_columns = [
            'workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'race', 'native-country']
        for column in categorical_columns:
            le = preprocessing.LabelEncoder()
            le.fit(features[column])
            train_feature[column] = pd.Series(le.transform(train_feature[column])).astype('category')
            test_feature[column] = pd.Series(le.transform(test_feature[column])).astype('category')
    else:
        raise ValueError('must not reach here!')

    # synthesis feature
    if synthesis:
        train_feature = synthesis_feature(train_feature)
        test_feature = synthesis_feature(test_feature)

    # feature selection
    if n_features is not None:
        selector = RFE(n_features_to_select=n_features)
        selector.fit(train_feature, train_y)
        train_feature = train_feature[train_feature.columns[selector.support_]]
        test_feature = test_feature[test_feature.columns[selector.support_]]
    return train_feature, test_feature


def synthesis_feature(data: pd.DataFrame) -> pd.DataFrame:
    es = ft.EntitySet(id="adult_dataset")
    es = es.entity_from_dataframe(
        entity_id='root',
        index='',
        dataframe=data,
        variable_types={
            'sex': ft.variable_types.Boolean,
        })
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_entity='root',
        trans_primitives=[
            'percentile', 'absolute',
        ])
    return feature_matrix
