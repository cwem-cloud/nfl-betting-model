import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import lightgbm as lgb

def train_ats(df, features):
    X = df[features]
    y = df["ats_home_cover"]
    model = XGBClassifier(
        max_depth=4, n_estimators=250, learning_rate=0.07, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
    )
    model.fit(X, y)
    return model

def train_ml(df, features):
    X = df[features]
    y = df["ml_home_win"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def train_totals(df, features):
    X = df[features]
    y = df["total_points"]
    model = lgb.LGBMRegressor(num_leaves=31, n_estimators=400, learning_rate=0.05)
    model.fit(X, y)
    return model
