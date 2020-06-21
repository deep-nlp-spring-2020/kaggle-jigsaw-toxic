
#%%

import os, time
import pandas as pd
import numpy as np

DATA_PATH =  "./data/"

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

#%%

from pathlib import Path

extra_files = [
    'extra.001.csv',
    'extra.002.csv',
    'extra.015.csv',
    'extra.019.csv',
]

for extra_file in extra_files:
    print(f"Processing {extra_file}...")
    df_extra  = pd.read_csv(os.path.join(DATA_PATH, extra_file))
    # print(df_extra.head())
    train_embeddings = calculate_embeddings(df_extra['comment_text'])
    model_name = f'models/{Path(extra_file).stem}.joblib'
    # print(model_name)
    joblib.dump(train_embeddings, model_name)


# %%
