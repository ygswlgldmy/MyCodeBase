from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import torch
import numpy as np
import pandas as pd

HOUSING_PATH = "2\housing"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    train_indices = shuffled_indices[test_set_size:]
    test_indices = shuffled_indices[:test_set_size]
    train, test = data[train_indices], data[test_indices]
    return train, test

def prepare_housing_data():

    housing = load_housing_data()

    housing_num = housing.drop("ocean_proximity", axis=1)
    # imputer = SimpleImputer(strategy="median")
    # imputer.fit(housing_num) # 需先计算出中位数才能使用transform来填充缺失值
    # XX = imputer.transform(housing_num)
    # housing_tr = pd.DataFrame(XX, columns=housing_num.columns, index=housing_num.index)

    odi_enc = OrdinalEncoder()
    housing_cat_encoded = odi_enc.fit_transform(housing[["ocean_proximity"]])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    # 分类特征
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OrdinalEncoder(), cat_attribs),
    ])

    return full_pipeline.fit_transform(housing)