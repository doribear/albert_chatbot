from tokenization_kbalbert import KbAlbertCharTokenizer

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

model_path = 'model'

tokenizer = KbAlbertCharTokenizer.from_pretrained(model_path)
vocab = tokenizer.get_vocab()
model = torch.load('30_20210817.pt')
model.cpu()
model.eval()
class_sheet = pd.read_excel('new_data/class_sheet.xlsx')


def get_similality(out_logits, tmp_answer):
    idx_answer = []
    for answer in tmp_answer:
        input = tokenizer(answer, return_tensors = 'pt', padding=True, truncation=True)
        out = model(**input)
        out = (out.logits - out_logits).mean()
        out = out.detach().numpy()
        out = np.abs(out)
        idx_answer.append(out)
    max = np.max(idx_answer)
    idx = idx_answer.index(max)
    return tmp_answer.iloc[1]

def chat(text):
    text = tokenizer(text, return_tensors = 'pt', padding=True, truncation=True)
    out = model(**text)
    out_logits = out.logits
    out = F.softmax(out_logits, dim = 1)
    label = out.argmax(dim = 1).item()
    tmp_answer = class_sheet.answer[class_sheet.label == label]
    if len(tmp_answer) > 1:
        answer = get_similality(out_logits, tmp_answer)
    else:
        answer = tmp_answer.iloc[0]
    return answer

if __name__ == '__main__':
    while True:
        text = input('입력 : ')
        print(chat(text))



