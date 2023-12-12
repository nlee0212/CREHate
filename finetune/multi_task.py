import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel,AutoModel,AutoTokenizer,AutoConfig,AutoModelForSequenceClassification
import logging
logging.basicConfig(level=logging.ERROR)
import os
from itertools import permutations


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

models = ['vinai/bertweet-base',
          './hate_bert',
          'Twitter/TwHIN-BERT-base',
          'cardiffnlp/twitter-roberta-base',
          'Xuhui/ToxDect-roberta-large',
          'bert-base-cased',
          'roberta-base']
model_names = [
    'BERTweet',
    'HateBERT',
    'TwHIN-BERT',
    'Twitter-RoBERTa',
    'ToxDect-RoBERTa',
    'BERT',
    'RoBERTa'
]
countries = ['United States','Australia','United Kingdom','South Africa','Singapore']
codes = ['US', 'AU', 'GB', 'ZA', 'SG']
_hate_cols = [f'{country.replace(" ","_")}_Hate' for country in countries]

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

class MultiTaskDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

class Classifier(torch.nn.Module):
    def __init__(self,model_name,tokenizer):
        super(Classifier, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name)
        self.l1.resize_token_embeddings(len(tokenizer))
        config = AutoConfig.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        
        self.classifier_1 = torch.nn.Linear(config.hidden_size, 2)
        self.classifier_2 = torch.nn.Linear(config.hidden_size, 2)
        self.classifier_3 = torch.nn.Linear(config.hidden_size, 2)
        self.classifier_4 = torch.nn.Linear(config.hidden_size, 2)
        self.classifier_5 = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        pooler = outputs[1]
        pooler = self.pre_classifier(pooler)
        pooler = self.dropout(pooler)
        output_1 = self.classifier_1(pooler)
        output_2 = self.classifier_2(pooler)
        output_3 = self.classifier_3(pooler)
        output_4 = self.classifier_4(pooler)
        output_5 = self.classifier_5(pooler)
        return output_1,output_2,output_3,output_4,output_5
    

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

def train(epoch,model,training_loader):
    model.train()
    loop = tqdm(enumerate(training_loader, 0),total=len(training_loader))
    loop.set_description(f"Epoch {epoch}")
    for _,data in loop:
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        output_1,output_2,output_3,output_4,output_5 = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss_1 = loss_fn(output_1, targets[:,0])
        loss_2 = loss_fn(output_2, targets[:,1])
        loss_3 = loss_fn(output_3, targets[:,2])
        loss_4 = loss_fn(output_4, targets[:,3])
        loss_5 = loss_fn(output_5, targets[:,4])
        loss = (loss_1 + loss_2 + loss_3 + loss_4 + loss_5)
        
        loop.set_postfix(loss=loss.item())
        
        loss.backward()
        optimizer.step()

def validation(testing_loader,model):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0),total=len(testing_loader)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            
            output_1,output_2,output_3,output_4,output_5 = model(ids, mask, token_type_ids)
            prob_1 = nn.Softmax(dim=1)(output_1)
            prob_2 = nn.Softmax(dim=1)(output_2)
            prob_3 = nn.Softmax(dim=1)(output_3)
            prob_4 = nn.Softmax(dim=1)(output_4)
            prob_5 = nn.Softmax(dim=1)(output_5)
                        
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs+=[[p1.cpu().detach().numpy().tolist(),p2.cpu().detach().numpy().tolist(),
                                 p3.cpu().detach().numpy().tolist(),p4.cpu().detach().numpy().tolist(),
                                 p5.cpu().detach().numpy().tolist()] for p1,p2,p3,p4,p5 in zip(prob_1, prob_2, prob_3, prob_4, prob_5)]

    return fin_outputs, fin_targets


MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 6
LEARNING_RATE = 2e-5
special_tokens = ["[US]","[AU]","[GB]","[ZA]","[SG]","@USER","URL"]

col_idx_permutation = list(permutations(range(5)))

for model_path,model_name in zip(models,model_names):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)
    tokenizer.add_tokens(special_tokens)
    
    res_row_list = []
    res_df = pd.DataFrame()
    
    train_file = './data_splits/CREHate_train.csv'
    valid_file = './data_splits/CREHate_valid.csv'
    test_file = './data_splits/CREHate_test.csv'
    
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    test_data = pd.read_csv(test_file)
    
    for idx,idx_permute in enumerate(col_idx_permutation):
        hate_cols = [_hate_cols[i] for i in idx_permute]
        
        train_df = pd.DataFrame()
        train_df['text'] = train_data['Text'] 
        train_df['labels'] = train_data[hate_cols].values.tolist()
        
        valid_df = pd.DataFrame()
        valid_df['text'] = valid_data['Text'] 
        valid_df['labels'] = valid_data[hate_cols].values.tolist()
        
        test_df = pd.DataFrame()
        test_df['text'] = test_data['Text'] 
        test_df['labels'] = test_data[hate_cols].values.tolist()
        
        
        training_set = MultiTaskDataset(train_df, tokenizer, MAX_LEN)
        valid_set = MultiTaskDataset(valid_df, tokenizer, MAX_LEN) 
        testing_set = MultiTaskDataset(test_df, tokenizer, MAX_LEN)
        
        train_params = {'batch_size': TRAIN_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': torch.cuda.device_count()
                        }
        valid_params = {'batch_size': VALID_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': torch.cuda.device_count()
                        }

        test_params = {'batch_size': VALID_BATCH_SIZE,
                        'shuffle': False,
                        'num_workers': torch.cuda.device_count()
                        }

        training_loader = DataLoader(training_set, **train_params)
        valid_loader = DataLoader(valid_set, **valid_params)
        testing_loader = DataLoader(testing_set, **test_params)
        
        model = Classifier(model_path,tokenizer)

        
        
        model = nn.DataParallel(model, device_ids =list(range(torch.cuda.device_count()))).to(device)
        
        optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        min_hamming_loss = 1
        best_model = None
        
        for epoch in range(EPOCHS):
            train(epoch,model,training_loader)
            outputs, targets = validation(valid_loader,model)

            final_outputs = np.array([[0 if output[0]>output[1] else 1 for output in row] for row in outputs])
            val_hamming_loss = metrics.hamming_loss(targets, final_outputs)
            val_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))
            print(f"Hamming Score = {val_hamming_score}")
            print(f"Hamming Loss = {val_hamming_loss}")
            
            if val_hamming_loss < min_hamming_loss:
                min_hamming_loss = val_hamming_loss
                best_model = model
                
        
        if best_model is not None:
            
            outputs, targets = validation(testing_loader,best_model)
            
            final_outputs = np.array([[0 if output[0]>output[1] else 1 for output in row] for row in outputs])
            
            tst_hamming_loss = metrics.hamming_loss(targets, final_outputs)
            tst_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))
            cols = [f'{model_name}-MT-{country}' for country in [codes[i] for i in idx_permute]]
            outputs_df = pd.DataFrame(final_outputs,columns=cols)
            total = pd.concat([test_data[hate_cols],outputs_df],axis=1)
            total.to_csv(f'./res/{model_name}-MT-ALL-P-{idx}-res.csv',index=False) 
            test_data  = pd.concat([test_data,outputs_df],axis=1)
            print(test_data)
            print(total)
            print('\tAcc\tF1\tH-F1\tN-F1')
            
            row = []
            for hate_col,code in zip(hate_cols,[codes[i] for i in idx_permute]):
                acc = metrics.accuracy_score(test_data[hate_col],outputs_df[f'{model_name}-MT-{code}'])
                f1 = metrics.f1_score(test_data[hate_col], outputs_df[f'{model_name}-MT-{code}'],average='macro')
                n,h = metrics.f1_score(test_data[hate_col], outputs_df[f'{model_name}-MT-{code}'],average=None)
                r = metrics.recall_score(test_data[hate_col], outputs_df[f'{model_name}-MT-{code}']) 
                print(f'{code}:\t{acc:.4f}\t{f1:.4f}\t{n:.4f}\t{h:.4f}\t{r:.4f}')
                row += [acc,f1,n,h,r]
            res_cols = []
            for code in [codes[i] for i in idx_permute]:
                res_cols += [f'{code}-{score}' for score in ['acc','f1','h','n','r']]
            res_df_row = pd.DataFrame([row],index=[idx],columns=res_cols)
            res_df = pd.concat([res_df,res_df_row])
            if 'avg' in res_df.index:
                res_df.drop('avg',inplace=True)
            res_df.loc['avg'] = res_df.mean(axis=0)
            print(res_df)
            res_df.to_csv(f'./res/{model_name}-MT-ALL-P-res-scores.csv')
