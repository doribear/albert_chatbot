from tokenization_kbalbert import KbAlbertCharTokenizer

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

model_path = 'model'

tokenizer = KbAlbertCharTokenizer.from_pretrained(model_path)
vocab = tokenizer.get_vocab()
model = torch.load('model.pt')
model.cpu()
model.eval()
class_sheet = pd.read_excel('data/class_sheet.xlsx')


def get_multichat(tmp_answer):
    answer_len = len(tmp_answer)
    idx_answer = np.random.randint(0, answer_len)
    return tmp_answer.iloc[idx_answer]

def chat(text):
    text = tokenizer(text, return_tensors = 'pt', padding=True, truncation=True)
    out = model(**text)
    out_logits = out.logits
    out = F.softmax(out_logits, dim = 1)
    label = out.argmax(dim = 1).item()
    tmp_answer = class_sheet.answer[class_sheet.label == label]
    if len(tmp_answer) > 1:
        answer = get_multichat(tmp_answer)
    else:
        answer = tmp_answer.iloc[0]
    return answer

if __name__ == '__main__':
    while True:
        text = input('입력 : ')
        print(chat(text))



