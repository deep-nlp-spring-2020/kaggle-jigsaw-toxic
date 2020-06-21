
#%%

import os, time
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score


# %%
SEED = 42
MAX_LENGTH = 224

DATA_PATH =  "./data/"
SST_TRAIN = "jigsaw-toxic-comment-train.csv"
SST_VALID = "validation.csv"
SST_TEST  = "test.csv"

# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# 
# Load and look at examples from [our first competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/). These are comments from Wikipedia with a variety of annotations (toxic, obscene, threat, etc).

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

import os
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib


#%%

train_embeddings = joblib.load('models/train_embeddings.joblib')
valid_embeddings = joblib.load('models/valid_embeddings.joblib')
test_embeddings  = joblib.load('models/test_embeddings.joblib')

#%%

train_labels = list(df_train['toxic'])
valid_labels = list(df_valid['toxic'])

# %%

class EmbeddingDataset(Dataset):

    def __init__(self, embeddings: list, labels: list=None):
        self.embeddings = embeddings
        self.labels = labels
        if labels:
            assert len(embeddings) == len(labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        if self.labels:
            return self.embeddings[index], self.labels[index]
        return self.embeddings[index]


# %%

class SimpleNNClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.cls_layer = nn.Linear(512, 1)

    def forward(self, embeddings):
        logits = self.cls_layer(embeddings)
        return logits

# %%

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def get_auc_from_logits(logits, labels):
    auc = -1
    try:
        probs = torch.sigmoid(logits.unsqueeze(-1))
        auc = roc_auc_score(labels.cpu().detach().numpy(), probs.squeeze().cpu().detach().numpy())
    except Exception as e:
        print(e)

    return auc

def evaluate(net, criterion, dataloader):
    net.eval()

    mean_acc, mean_auc, mean_loss = 0, 0, 0
    count = 0

    with torch.no_grad():
        for seq, labels in dataloader:
            seq, labels = seq.cuda(), labels.cuda()
            #seq, labels = seq, labels
            logits = net(seq)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            mean_auc += get_auc_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_auc / count, mean_loss / count


def train(net, criterion, opti, train_loader, val_loader, max_eps, print_every):
    net.train()
    best_auc = 0

    SAVE_PATH = 'models'
    os.makedirs(SAVE_PATH, exist_ok=True)

    for ep in range(max_eps):

        for it, (seq, labels) in enumerate(train_loader):
            opti.zero_grad()
            
            seq, labels = seq.cuda(), labels.cuda()
            # seq, labels = seq, labels

            logits = net(seq) #Obtaining the logits
            loss = criterion(logits.squeeze(-1), labels.float())
            loss.backward()
            opti.step()

            if (it + 1) % print_every == 0:
                #acc = get_accuracy_from_logits(logits, labels)
                auc = get_auc_from_logits(logits, labels)
                print(f"Iteration {it+1} of epoch {ep+1} complete. Loss : {loss.item()} AUC : {auc}")

        val_acc, val_auc, val_loss = evaluate(net, criterion, val_loader)
        print("Epoch {} complete! Validation AUC : {}, Validation Loss : {}".format(ep, val_auc, val_loss))
        if val_auc > best_auc:
            model_file_name = '{}/use_simple_nn_ep{}_auc{:.4f}_freeze.dat'.format(SAVE_PATH, ep, val_auc)
            print("Best validation auc improved from {} to {}, saving model '{}'...".format(best_auc, val_auc, model_file_name))
            torch.save(net.state_dict(), model_file_name)
            torch.save(net.state_dict(), f'{SAVE_PATH}/use_simple_nn_best.dat')
            best_auc = val_auc


# TODO CHECK
def predict(net, dataloader):
    net.eval()

    num_elements = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    predictions = torch.zeros(num_elements).cuda()
    for i, batch in enumerate(dataloader):
        batch = batch.cuda()
        start = i * batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        logits = net(batch)
        probs = torch.sigmoid(logits.flatten())
        predictions[start:end] = probs
    return predictions


#%%

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256

train_set = EmbeddingDataset(train_embeddings, train_labels)
val_set = EmbeddingDataset(valid_embeddings, valid_labels)

train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, num_workers=5, shuffle=True)  # TODO BUG SHUFFLE
val_loader = DataLoader(val_set, batch_size=EVAL_BATCH_SIZE, num_workers=5)

net = SimpleNNClassifier()
net.train()
net = net.cuda()
criterion = nn.BCEWithLogitsLoss()
opti = optim.Adam(net.parameters(), lr=1e-5)

#%%

# TODO BUG
net.load_state_dict(torch.load('models/use_simple_nn_best.dat'))

#%%

# TODO BUG

test_set = EmbeddingDataset(test_embeddings, None)
test_loader = DataLoader(test_set, batch_size=EVAL_BATCH_SIZE, num_workers=5)
preds = predict(net, test_loader).cpu().detach().numpy()
print("preds.shape", preds.shape)

test_df = pd.DataFrame()
test_df['toxic'] = preds
test_df.to_csv('use_sumbission.csv', index = None)

exit(0)


# %%

train(net, criterion, opti, train_loader, val_loader, 100, 200)

#%%
