from bert_model import BERTClassifier, EarlyStopping
import config
from datasets import make_dataloader
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, AdamW
import numpy as np
from metrics_for_multilabel import calculate_metrics, colwise_accuracy

DATA_PATH=config.DATA_PATH
model_config=config.model_config

class train():
    def __init__(self, device) -> None:
        # 데이터셋
        self.train_dataloader, self.test_dataloader = make_dataloader()
        self.device = device

        # KoBERT 라이브러리에서 bertmodel을 호출함. .to() 메서드는 모델 전체를 GPU 디바이스에 옮겨 줌.
        self.model = BERTClassifier(num_classes=config.num_calsses, dr_rate = model_config["dr_rate"]).to(self.device)

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


        # model_save_name = 'classifier'
        # model_file='.pt'
        # path = f"./bert_weights/{model_save_name}_{model_file}" 



    def train_model(self, batch_size, patience, n_epochs,path):
        
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 

        early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

        for epoch in range(1, n_epochs + 1):
            
            # initialize the early_stopping object
            self.model.train()
            train_epoch_pred=[]
            train_loss_record=[]

            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
            
                # label = label.long().to(device)
                label = label.float().to(self.device)

                out= self.model(token_ids, valid_length, segment_ids)#.squeeze(1)
                
                loss = self.loss_fn(out, label)

                train_loss_record.append(loss)

                train_pred=out.detach().cpu().numpy()
                train_real=label.detach().cpu().numpy()

                train_batch_result = calculate_metrics(np.array(train_pred), np.array(train_real))
                
                if batch_id%50==0:
                    print(f"batch number {batch_id}, train col-wise accuracy is : {train_batch_result['Column-wise Accuracy']}")
                    

                # save prediction result for calculation of accuracy per batch
                train_epoch_pred.append(train_pred)

                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), model_config["max_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule

                train_losses.append(loss.item())

            train_epoch_pred=np.concatenate(train_epoch_pred)
            train_epoch_target=self.train_dataloader.dataset.labels
            train_epoch_result=calculate_metrics(target=train_epoch_target, pred=train_epoch_pred)
            
            print(f"=====Training Report: mean loss is {sum(train_loss_record)/len(train_loss_record)}=====")
            print(train_epoch_result)
            
            print("=====train done!=====")

            # if e % log_interval == 0:
            #     print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))

            # print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
            test_epoch_pred=[]
            test_loss_record=[]

            self.model.eval()
            with torch.no_grad():
                for batch_id, (token_ids, valid_length, segment_ids, test_label) in enumerate(self.test_dataloader):
                    
                    token_ids = token_ids.long().to(self.device)
                    segment_ids = segment_ids.long().to(self.device)
                    valid_length = valid_length
                    
                    # test_label = test_label.long().to(device)
                    test_label = test_label.float().to(self.device)

                    test_out = self.model(token_ids, valid_length, segment_ids)

                    test_loss = self.loss_fn(test_out, test_label)

                    test_loss_record.append(test_loss)
                    
                    valid_losses.append(test_loss.item())

                    test_pred=test_out.detach().cpu().numpy()
                    test_real=test_label.detach().cpu().numpy()

                    test_batch_result = calculate_metrics(np.array(test_pred), np.array(test_real))

                    if batch_id%50==0:
                        print(f"batch number {batch_id}, test col-wise accuracy is : {test_batch_result['Column-wise Accuracy']}")

                    # save prediction result for calculation of accuracy per epoch
                    test_epoch_pred.append(test_pred)

            test_epoch_pred=np.concatenate(test_epoch_pred)
            test_epoch_target=self.test_dataloader.dataset.labels
            test_epoch_result=calculate_metrics(target=test_epoch_target, pred=test_epoch_pred)

            print(f"=====Testing Report: mean loss is {sum(test_loss_record)/len(test_loss_record)}=====")
            print(test_epoch_result)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(path))

        return self.model, avg_train_losses, avg_valid_losses
            

    def main(self):
        # early stopping patience; how long to wait after last time validation loss improved.
        patience = 10
        model, train_loss, valid_loss = self.train_model(model, 
                                                    model_config["batch_size"],
                                                    patience, 
                                                    model_config["num_epochs"], 
                                                    path=config.weight_path)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    train(device).main()