
#%%

import os, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score


# %%

DATA_PATH =  "./data/"
SST_TRAIN = "jigsaw-toxic-comment-train.csv"
SST_VALID = "validation.csv"
SST_TEST  = "test.csv"

# %%

df_train = pd.read_csv(os.path.join(DATA_PATH, SST_TRAIN))
df_train.head()

# %%
df_valid = pd.read_csv(os.path.join(DATA_PATH, SST_VALID))
df_valid.head()

# %%
df_test  = pd.read_csv(os.path.join(DATA_PATH, SST_TEST))
df_test.head()


# %%

import tensorflow_hub as hub
import tensorflow_text
from tqdm import tqdm
import joblib

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# %%

def calculate_embeddings(df_series):
    embeddings = []
    for _, row in tqdm(df_series.iteritems(), total=len(df_series)):
        embedding = use_model(row).numpy()[0]
        embeddings.append(embedding)

    return embeddings


# %%

train_embeddings = calculate_embeddings(df_train['comment_text'])
joblib.dump(train_embeddings, 'models/train_embeddings.joblib')

# %%

valid_embeddings = calculate_embeddings(df_valid['comment_text'])
joblib.dump(valid_embeddings, 'models/valid_embeddings.joblib')

# %%

test_embeddings = calculate_embeddings(df_test['content'])
joblib.dump(test_embeddings, 'models/test_embeddings.joblib')


# %%

# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression()
# clf.fit(train_embeddings, df_train['toxic'])
# roc_auc_score(df_valid['toxic'], clf.predict(valid_embeddings)),
# roc_auc_score(df_train['toxic'], clf.predict(train_embeddings)),
# roc_auc_score(df_train['toxic'], clf.predict(train_embeddings))


