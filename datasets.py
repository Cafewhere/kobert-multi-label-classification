from sklearn.model_selection import train_test_split
from bert_model import Data_for_BERT
import config
import torch


model_config=config.model_config


def make_dataloader(dataset):
    train_input, test_input = train_test_split(dataset, test_size = 0.1, random_state = 42)
    #print(train_input)
    train=train_input.copy()
    test=test_input.copy()

    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)

    data_train = Data_for_BERT(train, model_config["max_len"], True, False, x_cols = 'review', label_cols='label')
    data_test = Data_for_BERT(test, model_config["max_len"], True, False, x_cols = 'review', label_cols='label')

    # 파이토치 모델에 넣을 수 있도록 데이터를 처리함. 
    # data_train을 넣어주고, 이 테이터를 batch_size에 맞게 잘라줌. num_workers는 사용할 subprocess의 개수를 의미함(병렬 프로그래밍)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=model_config["batch_size"], num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=model_config["batch_size"], num_workers=0)

    return train_dataloader, test_dataloader