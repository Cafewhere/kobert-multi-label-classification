# config file for DeepVOC model

import os


DATA_PATH="./Data/dataset.csv"



def expand_pandas(max_rows=100, max_cols=500, width=None, max_info_cols=None):
    import pandas as pd
    if max_rows:
        pd.set_option("display.max_rows", max_rows) # 출력할 최대 행 갯수를 100으로 설정
    if max_cols:
        pd.set_option("display.max_columns", max_cols) # 출력할 최대 열 갯수를 500개로 설정
    if width:
        pd.set_option("display.width", width) # 글자 수 기준 출력할 넓이 설정
    if max_info_cols:
        pd.set_option("max_info_columns", max_info_cols) # 열 기반 info가 주어질 경우, 최대 넓이
    pd.set_option("display.float_format", lambda x : "%.3f" %x) # 출력할 float의 소숫점 자릿수 제한
    print("done")
    
    

model_config={"max_len" :512, 
              "batch_size":10,
             "warmup_ratio": 0.1,
             "num_epochs": 200,
             "max_grad_norm": 1,
             "learning_rate": 5e-6,
              "dr_rate":0.45}


label_cols=['음료',
            '디저트',
            '청결',
            '분위기',
            '대화',
            '공부',
            '가성비',
            '반려동물']

num_classes=8

weight_path="./weights"