import pickle

# Input Data 가공 파트
import pandas as pd
import numpy as np

import os
import re

import config
from config import expand_pandas

# KoBERT 모델
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook

from KoBERT.kobert.utils import get_tokenizer
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW

import warnings


DATA_PATH=config.DATA_PATH

warnings.filterwarnings("ignore")

num_class=8
weight_path="./weights/2023-05-09_08:35.pt"





class BERTClassifier(nn.Module):
        def __init__(self, bert, hidden_size = 768, num_classes = 8, dr_rate = None, params = None):
            # BERTClassifier의 자식 노드에 클래스 속성을 상속시켜준다?
            # 인풋으로 넣는 bert 모델을 클래스 내부의 bert 메서드로 넣어주고
            # dr_rate(drop-out rate)를 dr_rate로 넣어줌

            super(BERTClassifier, self).__init__()
            self.bert = bert
            self.dr_rate = dr_rate

            # 여기서 nn.Linear는 keras의 Dense 층과 같은 fully connected layer
            # hidden layer의 사이즈를 입력해주고(여기서는 768)
            # out-put layer의 사이즈를 num_classes 인자의 수만큼 잡아줌.
            # 아마 대/중/소분류 사이즈로 분리 가능할 듯.


    #         self.lstm_layer = nn.LSTM(512, 128, 2)
            self.classifier = nn.Linear(hidden_size, num_classes)

    #         self.classifier=Net(hidden_size=hidden_size, num_classes=num_classes)

            # dr_rate가 정의되어 있을 경우, 넣어준 비율에 맞게 weight를 drop-out 시켜줌
            if dr_rate:
                self.dropout = nn.Dropout(p=dr_rate)

        def generate_attention_mask(self, token_ids, valid_length):

            # 버트 모델에 사용할 attention_mask를 만들어 줌.
            # token_id를 인풋으로 받아, attention mask를 만들어 냄

            # torch.zeros_like()는 토치 텐서를 인풋으로 받아, 스칼라 값 0으로 채워진 같은 사이즈의 텐서를 뱉어냄
            attention_mask = torch.zeros_like(token_ids)

            for i,v in enumerate(valid_length):
                attention_mask[i][:v] = 1
            return attention_mask.float()

        def forward(self, token_ids, valid_length, segment_ids):
            #  attention mask 를 만들어 내고, 버트 모델을 넣어줌. 
            attention_mask = self.generate_attention_mask(token_ids, valid_length)

            # .long() pytorch는  .to()와 같은 기능을 수행함. 이는 장치(GPU)에 모델을 넣어주는 역할을 수행
            # 출력값으로 classifier()

            _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
            if self.dr_rate:
                out=self.dropout(pooler)

    #         output=self.lstm_layer(out)

            return self.classifier(out)
    



class load_model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower = False)
        self.transform = nlp.data.BERTSentenceTransform(self.tok, max_seq_length = config.model_config["max_len"], pad=True, pair=False)
        
        self.data=pd.read_csv('./Data/review_data.csv')

    def process_for_inference(self, num_of_text):
        text_input=self.data.iloc[num_of_text]['review']
        return text_input

    

    def get_model(self):
        # KoBERT 라이브러리에서 bertmodel을 호출함. .to() 메서드는 모델 전체를 GPU 디바이스에 옮겨 줌.
        self.model = BERTClassifier(self.bertmodel, num_classes=num_class, dr_rate = config.model_config["dr_rate"]).to(self.device)

        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval() 
        
        return self.model
    
    def get_prediction_from_txt(self, num_of_text):
        device=self.device
        
        input_text = self.process_for_inference(num_of_text)
        sentences = self.transform([input_text])
        
        get_pred = self.model(torch.tensor(sentences[0]).long().unsqueeze(0).to(device),torch.tensor(sentences[1]).unsqueeze(0),torch.tensor(sentences[2]).to(device))
        get_pred = torch.sigmoid(get_pred).squeeze()
        
        get_pred = (get_pred >= 0.5).nonzero(as_tuple=True)[0]
        
        result=f"분석 결과, 예상 태그는 {[config.label_cols[i] for i in get_pred]} 입니다."
        return result

    def make_prediction_from_idx(self, start_idx, end_idx=-1):
        if end_idx == -1:
            end_idx = start_idx+1

        device=self.device
        
        for num_of_text in range(start_idx, end_idx):
            
            input_text = self.process_for_inference(num_of_text)
            print(f'\n[{num_of_text}] : {input_text}')
            sentences = self.transform([input_text])
            
            get_pred = self.model(torch.tensor(sentences[0]).long().unsqueeze(0).to(device),torch.tensor(sentences[1]).unsqueeze(0),torch.tensor(sentences[2]).to(device))
            #print(get_pred)
            get_pred = (torch.sigmoid(get_pred).squeeze() >= 0.5).int()
            #print(get_pred)
            pred = (get_pred >= 0.5).nonzero(as_tuple=True)[0]
        
            result=f" => 분석 결과, 예상 태그는 {[config.label_cols[i] for i in pred]} 입니다."
            print(result)
        
        
        
        
        
        

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    model=load_model()
    bert=model.get_model()
    bert.eval()

    
    #print(model.get_prediction_from_txt(0))
    model.make_prediction_from_idx(20, 25)
        