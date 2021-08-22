from tokenization_kbalbert import KbAlbertCharTokenizer
from transformers import AlbertForSequenceClassification, AdamW

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from tqdm import tqdm

from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'model'

tokenizer = KbAlbertCharTokenizer.from_pretrained(model_path)
vocab = tokenizer.get_vocab()
#model = AlbertForSequenceClassification.from_pretrained(model_path)

model = torch.load('20210817_1.pt')

class bertset(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, label):
        super(bertset, self).__init__()
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label = label

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


data = pd.read_excel('new_data/train_data_aug.xlsx')

tokenized = tokenizer(list(data.text), return_tensors = 'pt', padding=True, truncation=True)
input_ids = tokenized.input_ids
token_type_ids = tokenized.token_type_ids
attention_mask = tokenized.attention_mask
label = torch.tensor(list(data.label))

trainset = bertset(input_ids, token_type_ids, attention_mask, label)


data = pd.read_excel('new_data/daily_test.xlsx')

tokenized = tokenizer(list(data.text), return_tensors = 'pt', padding=True, truncation=True)
input_ids = tokenized.input_ids
token_type_ids = tokenized.token_type_ids
attention_mask = tokenized.attention_mask
label = torch.tensor(list(data.label))

valset = bertset(input_ids, token_type_ids, attention_mask, label)

trainloader = DataLoader(trainset, batch_size = 32, shuffle = True)
valloader = DataLoader(valset, batch_size = 32, shuffle = True)

optimizer = AdamW(model.classifier.parameters(), lr = 0.001)
#optimizer = optim.Adam(model.classifier.parameters(), lr = 0.01)

train_accuracies = []
val_accuracies = []
train_lossese = []
val_lossese = []


for epoch in range(300):
    model.train()
    model.to(device)


    processor = tqdm(trainloader)
    accuracies = []
    losses = []
    for idx, (input_ids, token_type_ids, attention_mask, label) in enumerate(processor):
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)

        outs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = label)
        loss = outs.loss
        loss.backward()
        optimizer.step()
        model_out = outs.logits
        model_out = F.softmax(model_out, dim = 1)
        model_out = model_out.argmax(dim = 1)
        accuracy = accuracy_score(model_out.cpu(), label.cpu())
        accuracies.append(accuracy)
        losses.append(loss.cpu().item())
        if idx % 50 == 0:
            print(f'epoch : {epoch}, idx : {idx}, loss = {loss.cpu().item()}, accuracy : {accuracy}')
        del input_ids, token_type_ids, attention_mask, label
    train_accuracy = np.mean(accuracies)
    train_accuracies.append(train_accuracy)
    loss = np.mean(losses)
    train_lossese.append(loss)
    print(f'TrainSet :: epoch : {epoch}, accuracy : {train_accuracy}, loss : {loss}')

    processor = tqdm(valloader)
    accuracies = []
    losses = []
    for idx, (input_ids, token_type_ids, attention_mask, label) in enumerate(processor):
        model.cpu()
        model.eval()
        outs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = label)
        model_out = outs.logits
        model_out = F.softmax(model_out, dim = 1)
        model_out = model_out.argmax(dim=1)
        loss = outs.loss
        accuracy = accuracy_score(model_out.cpu(), label.cpu())
        accuracies.append(accuracy)
        losses.append(loss.cpu().item())
    val_accuracy = np.mean(accuracies)
    val_accuracies.append(val_accuracy)
    loss = np.mean(losses)
    val_lossese.append(loss)
    print(f'EvalSet :: epoch : {epoch}, accuracy : {val_accuracy}, loss : {loss}')
    if train_accuracy >= 0.95 and val_accuracy >= 0.95:
        print('accuracy is over 0.95 so we stop training!!')

    if epoch % 10 == 0:
        torch.save(model, f'{epoch}_20210817_1.pt')

history = pd.DataFrame({'train_accuracies' : train_accuracies, 'train_lossese' : train_lossese,
                        'val_accuracies' : val_accuracies, 'val_lossese' : val_lossese})

history.to_excel('history_new.xlsx', index = False)
torch.save(model, '20210817_1.pt')
