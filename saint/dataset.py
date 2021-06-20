from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class Preprocess:
    def __init__(self, train_data, test_data, max_len=200):
        self.padding = 0
        self.max_len = max_len
        self.train_data = train_data.copy()
        # self.valid_data = valid_data
        self.test_data = test_data.copy()
        self.enc_item = LabelEncoder()
        self.enc_test = LabelEncoder()
        self.enc_knowledge = LabelEncoder()
        self.process()
        self.train_data, self.valid_data = self.split()
    
    def split(self):
        train_uid, valid_uid = train_test_split(self.train_data['userID'].unique(), random_state=156, test_size=0.1)
        train_, valid_ = self.train_data[self.train_data['userID'].isin(train_uid)], self.train_data[self.train_data['userID'].isin(valid_uid)]
        return train_, valid_
    
    def process(self):
        # Label encode
        self.train_data['assessmentItemID'] = self.enc_item.fit_transform(self.train_data['assessmentItemID'])
        #self.valid_data['assessmentItemID'] = self.enc_item.transform(self.valid_data['assessmentItemID'])
        self.test_data['assessmentItemID'] = self.enc_item.transform(self.test_data['assessmentItemID'])
        self.train_data['testId'] = self.enc_test.fit_transform(self.train_data['testId'])
        #self.valid_data['testId'] = self.enc_test.transform(self.valid_data['testId'])
        self.test_data['testId'] = self.enc_test.transform(self.test_data['testId'])
        self.train_data['KnowledgeTag'] = self.enc_knowledge.fit_transform(self.train_data['KnowledgeTag']) 
        #self.valid_data['KnowledgeTag'] = self.enc_knowledge.transform(self.valid_data['KnowledgeTag'])
        self.test_data['KnowledgeTag'] = self.enc_knowledge.transform(self.test_data['KnowledgeTag'])
        self.train_data = self.train_data.groupby('userID').tail(self.max_len)
        self.test_data = self.test_data.groupby('userID').tail(self.max_len)
        
    def create_dataset(self):
        train_ = self.train_data.groupby('userID').apply(lambda x: self._grouper(x)).to_list()
        valid_ = self.valid_data.groupby('userID').apply(lambda x: self._grouper(x)).to_list()
        test_ = self.test_data.groupby('userID').apply(lambda x: self._grouper(x)).to_list()
        return train_, valid_, test_
        
    def _grouper(self, x):
        userID = x['userID'].values
        assessmentItemID = list(x['assessmentItemID'].values)
        testId = list(x['testId'].values)
        answerCode = list(x['answerCode'].values)
        KnowledgeTag = list(x['KnowledgeTag'].values)
        length = len(answerCode)
        
        if length < self.max_len: # Pad
            assessmentItemID = [0]*(self.max_len - length) + assessmentItemID
            testId = [0]*(self.max_len - length) + testId
            answerCode = [0]*(self.max_len - length) + answerCode
            KnowledgeTag = [0]*(self.max_len - length) + KnowledgeTag
#             padded = (self.max_len - length) * [True] + length * [False]
            padded = (self.max_len - length) * [True] + length * [False]
        
        else: # Limit
            assessmentItemID = assessmentItemID[length - (self.max_len):]
            testId = testId[length - (self.max_len):]
            answerCode = answerCode[length - (self.max_len):]
            KnowledgeTag = KnowledgeTag[length - (self.max_len):]
            padded = self.max_len * [False]
        
        result = {
            'userID': userID[-1],
            'assessmentItemID': assessmentItemID,
            'testId': testId,
            'answerCode': answerCode,
            'KnowledgeTag': KnowledgeTag,
            'padded': padded
        }
        return result

class SAINTDataset:
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return idx, self.data[idx]['assessmentItemID'], self.data[idx]['testId'], self.data[idx]['KnowledgeTag'], self.data[idx]['padded'], self.data[idx]['answerCode']
    
def collate_fn(batch):
    _, assessmentItemID, testId, KnowledgeTag, padded, labels = zip(*batch)
    assessmentItemID = torch.Tensor(assessmentItemID).long()
    testId = torch.Tensor(testId).long()
    KnowledgeTag = torch.Tensor(KnowledgeTag).long()
    padded = torch.Tensor(padded).bool()
    labels = torch.Tensor(labels)
    return assessmentItemID, testId, KnowledgeTag, padded, labels