# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from sklearnex import patch_sklearn
patch_sklearn()


# %%
import pandas as pd
import numpy as np
from time import time
from sklearn.datasets import load_digits


# %%
data = pd.read_csv("datasets/income.csv")
data


# %%
data["education"] = data["education"].replace(
    {
        'Preschool': 'dropout',
        '10th': 'dropout',
        '11th': 'dropout',
        '12th': 'dropout',
        '1st-4th': 'dropout',
        '5th-6th': 'dropout',
        '7th-8th': 'dropout',
        '9th': 'dropout',
        'HS-Grad': 'HighGrad',
        'HS-grad': 'HighGrad',
        'Some-college': 'CommunityCollege',
        'Assoc-acdm': 'CommunityCollege',
        'Assoc-voc': 'CommunityCollege',
        'Bachelors': 'Bachelors',
        'Masters': 'Masters',
        'Prof-school': 'Masters',
        'Doctorate': 'Doctorate'
    }
)
data["marital-status"] = data["marital-status"].replace(
    {
        'Never-married': 'NotMarried',
        'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married',
        'NotMarried': 'NotMarried',
        'Separated': 'Separated',
        'Divorced': 'Separated',
        'Widowed': 'Widowed',
    }
)
ordinal_columns={
    "education": ["dropout", "HighGrad", "CommunityCollege", "Bachelors", "Masters", "Doctorate"],
}
data["native-country"] = data["native-country"].replace(
{"Canada": "North America", "Cuba": "South America", "Dominican-Republic": "South America", "El-Salvador": "South America", "Guatemala": "South America",
                   "Haiti": "South America", "Honduras": "South America", "Jamaica": "South America", "Mexico": "South America", "Nicaragua": "South America",
                   "Outlying-US(Guam-USVI-etc)": "North America", "Puerto-Rico": "North America", "Trinadad&Tobago": "South America",
                   "United-States": "North America", "Cambodia": "Asia", "China": "Asia", "Hong": "Asia", "India": "Asia", "Iran": "Asia", "Japan": "Asia", "Laos": "Asia",
          "Philippines": "Asia", "Taiwan": "Asia", "Thailand": "Asia", "Vietnam": "Asia", "Columbia": "South America", "Ecuador": "South America", "Peru": "South America", "England": "West Europe", "France": "West Europe", "Germany": "West Europe", "Greece": "West Europe", "Holand-Netherlands": "West Europe",
            "Ireland": "West Europe", "Italy": "West Europe", "Portugal": "West Europe", "Scotland": "West Europe", "Hungary": "East Europe", "Poland": "East Europe", "Yugoslavia": "East Europe", "South": "?"})
data["capital"] = data["capital-gain"]-data["capital-loss"]
data.drop(["capital-gain", "capital-loss"], axis=1, inplace=True)
data["education-num"] = data["education-num"].astype(float)


# %%
from sklearn.model_selection import RepeatedStratifiedKFold
from automl.search.automl import AutoML

am = AutoML(
    random_state=42,
    level="uncommon",
    target_metric="f1",
    trainer_config={
        #"secondary_level": ComponentLevel.UNCOMMON,
        "cache": True,
        "early_stopping": True,
        "return_test_scores_during_tuning": True,
        "tuning_time": 200*3,
        "stacking_level": 3,
        "tune_kwargs": {"max_concurrent": 2, "trainable_n_jobs": 4}
    },
    #cv=RepeatedStratifiedKFold(random_state=420, n_repeats=2, n_splits=4)
    cv=4,
)


# %%
t_s = time()
am.fit(data.drop("income >50K", axis=1), data["income >50K"], ordinal_columns={"education":ordinal_columns})
#am.fit(data.drop("Purchase", axis=1), data["Purchase"])
#am.fit(data.drop("default", axis=1), data["default"], ordinal_columns={"EDUCATION": list(sorted(data["EDUCATION"].unique()))})
#am.fit(data.drop("quality", axis=1), data["quality"]>=6)
#am.fit(data.drop("target", axis=1), data["target"])
#am.fit(data.drop("target", axis=1), data["target"], ordinal_columns=ordinal_columns)
#am.fit(X_train, y_train, X_test=X_test, y_test=y_test)
t_end = time() - t_s


# %%
print(am.results_)
