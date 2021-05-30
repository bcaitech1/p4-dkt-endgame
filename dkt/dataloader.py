import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        cols_name = [col_name for col_name in df.columns if col_name[:4] == 'add_']

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)


        for col in cols_name:
            if col[-4:] == '_cat':
                le = LabelEncoder()
                if is_train:
                    # For UNKNOWN class
                    a = df[col].unique().tolist() + ['unknown']
                    le.fit(a)
                    self.__save_labels(le, col)
                else:
                    label_path = os.path.join(self.args.asset_dir, col + '_classes.npy')
                    le.classes_ = np.load(label_path)

                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

                df[col] = df[col].astype(str)
                test = le.transform(df[col])
                df[col] = test

        df['Timestamp'] = df['Timestamp'].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        #0529 추가 feature는 column명에 'add_'를 붙여주세요.
        #     categorical이면 '_cat', continuous면 '_con'을 뒤에 붙여주세요.
        data_type = {'userID': object, 'KnowledgeTag': object, 'answerCode': 'int16'}
        df = df.astype(data_type)

        df['add_test_pre3_cat'] = df.assessmentItemID.map(lambda x: x[1:4])
        df['add_test_post3_cat'] = df.assessmentItemID.map(lambda x: x[4:7])
        df['add_test_item_cat'] = df.assessmentItemID.map(lambda x: x[-3:])
        df['add_knowledgetag_con'] = df.KnowledgeTag.astype(np.float32)
        ###
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)#, nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        #0529 categorical feature와 continuous feature를 list로 저장합니다.
        self.args.cate_cols = [col for col in df.columns if col[:4] == 'add_' and col[-4:] == '_cat']
        self.args.cont_cols = [col for col in df.columns if col[:4] == 'add_' and col[-4:] == '_con']

        #0529 total_cate_size: feature를 merge할때를 위해 categorical feature의 가짓수를 전부 저장합니다.
        #     n_cate_size    : 각 categorical feature의 가짓수를 저장합니다.
        #     cate_idx       : categorical feature의 index를 저장합니다.
        self.args.total_cate_size = 1
        self.args.n_cate_size = {}
        self.args.cate_idx = {}
        for i, col in enumerate(self.args.cate_cols):
            self.args.total_cate_size += len(np.load(os.path.join(self.args.asset_dir, col + '_classes.npy')))
            self.args.n_cate_size[col] = len(np.load(os.path.join(self.args.asset_dir, col + '_classes.npy')))
            self.args.cate_idx[i] = col
        print(self.args.n_cate_size)
        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)

        #0529 feature를 merge한다면 dataframe을 반환합니다.
        #     merge하지 않는다면 userID로 grouping하여 그 value를 반환합니다. (baseline)
        if self.args.merge_feature:
            return df
        else:
            columns = ['userID', 'answerCode'] + self.args.cate_cols + self.args.cont_cols
            group = df[columns].groupby('userID').apply(
                lambda r: tuple(
                    r[col_name].values for col_name in columns[1:]
                )
            )
            return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DKTDataset_merge(torch.utils.data.Dataset):
    #0531 merge시 사용하는 dataset입니다.
    #     dataframe을 입력으로 받습니다.
    #     categorical column과 continuous column, correct를 나눠 grouping하여 저장합니다.
    def __init__(self, df, max_seq_len=40):
        self.df = df
        self.max_seq_len = max_seq_len
        self.cate_cols = []
        self.cont_cols = []
        for col in df.columns:
            if col[-4:] == '_cat':
                self.cate_cols.append(col)
            elif col[-4:] == '_con':
                self.cont_cols.append(col)

        self.correct_seq = self.df[['userID', 'answerCode']].groupby('userID').apply(
            lambda r: torch.tensor(tuple(
                r['answerCode'].values
            ))
        ).values
        tmp = ['userID'] + self.cate_cols
        self.cate_seq = self.df[tmp].groupby('userID').apply(
            lambda r: torch.tensor(tuple(
                r[col_name].values for col_name in self.cate_cols
            ))
        ).values
        tmp = ['userID'] + self.cont_cols
        self.cont_seq = self.df[tmp].groupby('userID').apply(
            lambda r: torch.tensor(tuple(
                r[col_name].values for col_name in self.cont_cols
            ))
        ).values

    def __getitem__(self, index):
        #0531 max_seq_len으로 내용을 뒤에서부터 잘라 저장하여 반환합니다.
        correct_seq = self.correct_seq[index]
        cate_seq = self.cate_seq[index]
        cont_seq = self.cont_seq[index]
        seq_len = len(correct_seq)
        if seq_len > self.max_seq_len:
            seq_len = self.max_seq_len

        cate_feature = torch.zeros(self.max_seq_len, len(self.cate_cols), dtype=torch.long)
        cont_feature = torch.zeros(self.max_seq_len, len(self.cont_cols), dtype=torch.float)
        mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
        target = torch.zeros(self.max_seq_len)

        if len(self.cate_cols) != 0:
            cate_feature[-seq_len:] = cate_seq[:, -seq_len:].T.clone().detach()
        if len(self.cont_cols) != 0:
            cont_feature[-seq_len:] = cont_seq[:, -seq_len:].T.clone().detach()
        mask[-seq_len:] = 1
        target[-seq_len:] = torch.ShortTensor(correct_seq[-seq_len:])
        return cate_feature, cont_feature, mask, target

    def __len__(self):
        return len(self.correct_seq)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]
        seq_len = len(row[0])

        cols = list(row)
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cols):
                cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        cols.append(mask)


        for i, col in enumerate(cols):
            cols[i] = torch.tensor(col)

        return cols


    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)


    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):
    pin_memory = False
    train_loader, valid_loader = None, None
    #0531 merge라면 dataframe, 그렇지않다면 group.values로 입력받습니다.
    if args.merge_feature:
        #0531 max_seq_len으로 선언하여 사용하기때문에 collate 함수는 사용하지 않습니다.
        if train is not None:
            trainset = DKTDataset_merge(train, args.max_seq_len)
            train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                                                   batch_size=args.batch_size, pin_memory=pin_memory)
        if valid is not None:
            valset = DKTDataset_merge(valid, args.max_seq_len)
            valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=True,
                                                   batch_size=args.batch_size, pin_memory=pin_memory)
    else:
        if train is not None:
            trainset = DKTDataset(train, args)
            train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                                batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
        if valid is not None:
            valset = DKTDataset(valid, args)
            valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                                batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader
