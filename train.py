from bert_model import BERTClassifier, EarlyStopping
import config
from datasets import make_dataloader
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, AdamW
import numpy as np
from metrics_for_multilabel import calculate_metrics, colwise_accuracy
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

DATA_PATH=config.DATA_PATH
model_config=config.model_config

class train():
    def __init__(self, device) -> None:
        # 데이터셋
        self.train_dataloader, self.test_dataloader = make_dataloader(pd.read_csv(DATA_PATH, index_col=0))
        self.device = device

        # KoBERT 라이브러리에서 bertmodel을 호출함. .to() 메서드는 모델 전체를 GPU 디바이스에 옮겨 줌.
        self.model = BERTClassifier(num_classes=config.num_classes, dr_rate = model_config["dr_rate"]).to(self.device)

        # 옵티마이저와 스케쥴 준비 (linear warmup과 decay)
        no_decay = ['bias', 'LayerNorm.weight']

        # no_decay에 해당하는 파라미터명을 가진 레이어들은 decay에서 배제하기 위해 weight_decay를 0으로 셋팅, 그 외에는 0.01로 decay
        # weight decay란 l2 norm으로 파라미터 값을 정규화해주는 기법을 의미함
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


        # 옵티마이저는 AdamW, 손실함수는 BCE
        # optimizer_grouped_parameters는 최적화할 파라미터의 그룹을 의미함
        self.optimizer = AdamW(optimizer_grouped_parameters, lr= model_config["learning_rate"])
        # loss_fn = nn.CrossEntropyLoss()
        self.loss_fn=nn.BCEWithLogitsLoss()


        # t_total = train_dataloader.dataset.labels.shape[0] * num_epochs
        # linear warmup을 사용해 학습 초기 단계(배치 초기)의 learning rate를 조금씩 증가시켜 나가다, 어느 지점에 이르면 constant하게 유지
        # 초기 학습 단계에서의 변동성을 줄여줌.

        t_total = len(self.train_dataloader) * model_config["num_epochs"]
        warmup_step = int(t_total * model_config["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        self.file_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
        # model_save_name = 'classifier'
        # model_file='.pt'
        # path = f"./bert_weights/{model_save_name}_{model_file}" 



    def train_model(self, batch_size, patience, n_epochs, path):
        writer = SummaryWriter(f'runs/{self.file_name}')
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=f'{path}/{self.file_name}.pt')

        for epoch in range(1, n_epochs + 1):
            print(f"========= {epoch} epoch ==================")
            # initialize the early_stopping object
            self.model.train()
            train_loss = 0
            for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(self.train_dataloader), total = len(self.train_dataloader)):
                self.optimizer.zero_grad()

                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
            
                label = label.float().to(self.device)

                out= self.model(token_ids, valid_length, segment_ids)#.squeeze(1)
                # print(f'out:\n{out}')
                # print(f'label:\n{label}')
                
                loss = self.loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), model_config["max_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule

                train_loss += loss.item()
            train_loss = train_loss / len(self.train_dataloader)
            writer.add_scalar("loss(train)", train_loss, epoch) # tensorboard
            print(f"train loss: {train_loss}")


            self.model.eval()
            valid_loss = 0
            valid_accuracy = 0
            with torch.no_grad():
                cidx = np.random.randint(len(self.test_dataloader))
                for batch_id, (token_ids, valid_length, segment_ids, test_label) in tqdm(enumerate(self.test_dataloader), total = len(self.test_dataloader)):
                    
                    token_ids = token_ids.long().to(self.device)
                    segment_ids = segment_ids.long().to(self.device)
                    valid_length = valid_length
                    
                    # test_label = test_label.long().to(device)
                    test_label = test_label.float().to(self.device)

                    test_out = self.model(token_ids, valid_length, segment_ids)

                    test_loss = self.loss_fn(test_out, test_label)
                    
                    valid_loss += test_loss.item()

                    test_real=np.array(test_label.detach().cpu().numpy())
                    test_pred=np.array((torch.sigmoid(test_out).detach().cpu().numpy()>=0.5).astype(int))
                    # print(f'test_pred:\n{test_pred}')
                    # print(f'test_real:\n{test_real}')
                    valid_accuracy += colwise_accuracy(test_real, test_pred)
                
            valid_loss = valid_loss / len(self.test_dataloader)
            valid_accuracy = valid_accuracy / len(self.test_dataloader)
            writer.add_scalar("loss(valid)", valid_loss, epoch) # tensorboard
            writer.add_scalar("accuracy(valid)", valid_accuracy, epoch) # tensorboard
            print(f"valid loss: {valid_loss}")
            print(f"colwise_mean_accuracy: {valid_accuracy}")

            

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(path))

        return self.model
            

    def main(self):
        # early stopping patience; how long to wait after last time validation loss improved.
        patience = 10
        model  = self.train_model(
                            model_config["batch_size"],
                            patience, 
                            model_config["num_epochs"], 
                            path=config.weight_path) # batch_size, patience, n_epochs, path

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    train(device).main()