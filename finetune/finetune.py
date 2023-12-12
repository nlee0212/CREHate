# Standard
import os
import sys
import shutil
import argparse

# PIP
from tqdm import tqdm
import random
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5Tokenizer,T5ForConditionalGeneration 
from transformers import AdamW, get_linear_schedule_with_warmup

# Custom
from config import Config
from dataset import CSV2Dataset, batch_sampling

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, help='Config file name')
parser.add_argument('--model', type=str, default=None, help='Model name')
parser.add_argument('--checkpoint_filename', type=str, default=None)
parser.add_argument('--best_filename', type=str, default=None)

parser.add_argument('--train_data', type=str, default=None)
parser.add_argument('--val_data', type=str, default=None,
                    help='Train Label column name')
parser.add_argument('--test_data', type=str, default=None,
                    help='Train Label column name')
parser.add_argument('--test_res', type=str, default=None,
                    help='Train Label column name')

parser.add_argument('--sent_col', type=str, )
parser.add_argument('--label_col', type=str, default=None,
                    help='Train Label column name')
parser.add_argument('--num_labels', type=str, default=None)

parser.add_argument('--test_res_col', type=str, default=None,
                    help='Test result column name')
parser.add_argument('--test_col', type=str,)

parser.add_argument('--train', type=str, default=None,
                    help='Train Label column name')
parser.add_argument('--evaluate', type=str, default=None,
                    help='Train Label column name')
parser.add_argument('--test', type=str, default=None,
                    help='Train Label column name')

parser.add_argument('--resume', type=str, default=None,
                    help='Train Label column name')
parser.add_argument('--resume_model', type=str, default=None,
                    help='Train Label column name')

args = parser.parse_args()


def save_checkpoint(state, is_best, filename, best_filename):
    if is_best:
        print('Best F1 Updated -- Saving Best Checkpoint')
        torch.save(state, best_filename)

def train(train_loader, model, optimizer, scheduler, epoch, epochs):
    model.train()
    
    total_loss, total_accuracy = 0, 0
    print("-"*30)

    loop = tqdm(train_loader, leave=True)
    for (input_ids,attention_mask),labels in loop:
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        
        if 't5' in cfg.model:
            lm_label = labels[0].to(device)
            label_mask = labels[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=lm_label, decoder_attention_mask = label_mask)
        else: 
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
        
        loss = outputs.loss.mean()
        total_loss += loss.item()

        outputs.loss.mean().backward()
        


        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        
        optimizer.step()
        scheduler.step()
        
    avg_loss = total_loss / len(train_loader)
    print(f" {epoch+1} Epoch Average train loss :  {avg_loss}")

def validate(valid_loader, model, tokenizer):
    model.eval()

    total_true = []
    total_pred = []
    real_total_true = []
    real_total_pred = []

    loop = tqdm(valid_loader, leave=True)
    for (input_ids,attention_mask),labels in loop:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)


        with torch.no_grad():
            if 't5' in cfg.model:
                lm_label = labels[0].to(device)
                label_mask = labels[1].to(device)
                outputs = model.module.generate(input_ids, attention_mask=attention_mask,max_length=3)
                pred = [tokenizer.decode(label) for label in outputs]
                pred = [l.replace('<pad>','').replace(' ','')[0] for l in pred]
                try:
                    _pred = [int(l) for l in pred]
                    pred = _pred
                except:
                    _pred = []
                    for l in pred:
                        if l == '0' or l == '1':
                            _pred.append(int(l))
                        else:
                            _pred.append(l)
                    pred = _pred
                

                lm_label[lm_label[:, :] == -100] = tokenizer.pad_token_id
                true = [tokenizer.decode(label) for label in lm_label]
                true = [int(l[0]) for l in true]

                _pred = []
                _true = []

                for i in range(len(true)):
                    if type(pred[i]) is int:
                        _pred.append(pred[i])
                        _true.append(true[i])

                
            else: 
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask,
                                labels=labels)

                pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
                
    
                true = [label for label in labels.cpu().numpy()]
                
            
        total_true += true
        total_pred += pred

        if 't5' in cfg.model:
            real_total_true += _true
            real_total_pred += _pred
        
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(total_true, total_pred, average='macro')
        accuracy = accuracy_score(total_true, total_pred)

        print(f"validation accuracy : {accuracy: .4f}")
        print(f"validation Precision : {precision: .4f}")
        print(f"validation Recall : {recall: .4f}")
        print(f"validation F1 : {f1: .4f}")
    except:
        precision, recall, f1, _ = precision_recall_fscore_support(real_total_true, real_total_pred, average='macro')
        accuracy = accuracy_score(real_total_true, real_total_pred)

        print(f"No. of int type output: {len(real_total_true)}/{len(total_true)}")
        print(f"validation accuracy : {accuracy: .4f}")
        print(f"validation Precision : {precision: .4f}")
        print(f"validation Recall : {recall: .4f}")
        print(f"validation F1 : {f1: .4f}") 
    
    return f1

def test(test_loader,model,tokenizer):
    model.eval()

    final_preds = []
    idss = []

    loop = tqdm(test_loader, leave=True)
    for input_ids,attention_mask in loop:

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        argmax = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
        print(argmax)
        final_preds += argmax
        idss += input_ids.cpu()

    print(len(idss))
    sents = [tokenizer.decode(ids) for ids in idss]
    print(len(sents))
    
    print(len(final_preds))
    
    return (sents,final_preds)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cfg = Config(args.config_file)

for key, value in vars(args).items():
    if value != None:
        exec("cfg.%s = '%s'" % (key,value))


print(cfg.model, cfg.sent_col, cfg.label_col)

cfg.num_workers = torch.cuda.device_count()
print(cfg.test_res_col)

print(cfg.checkpoint_filename)

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

print('Train:',cfg.train)
print('Evaluate:',cfg.evaluate)
print('Test:',cfg.test)

if 't5' in cfg.model:
    model = T5ForConditionalGeneration.from_pretrained(cfg.model,resume_download=True)
else:
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model,resume_download=True)


optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

if cfg.train:
    train_dataset = CSV2Dataset(cfg,cfg.train_data,'train')
    valid_dataset = CSV2Dataset(cfg,cfg.val_data,'valid')
    train_dataloader = DataLoader(train_dataset,batch_sampler=batch_sampling(cfg.batch_size,len(train_dataset)))
    valid_dataloader = DataLoader(valid_dataset,batch_sampler=batch_sampling(cfg.batch_size,len(valid_dataset)))
    if cfg.evaluate:
        test_dataset = CSV2Dataset(cfg,cfg.test_data,'valid') 
        test_dataloader = DataLoader(test_dataset,batch_sampler=batch_sampling(cfg.batch_size,len(test_dataset)))
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0, 
                                                num_training_steps=len(train_dataloader)*cfg.epochs)
    tokenizer = valid_dataset.get_tokenizer()

elif cfg.test:
    test_dataset = CSV2Dataset(cfg,cfg.test_data,'test')
    test_dataloader = DataLoader(test_dataset,batch_sampler=batch_sampling(cfg.batch_size,len(test_dataset),is_test=True))
    tokenizer = test_dataset.get_tokenizer()

if cfg.evaluate and not cfg.train:
    valid_dataset = CSV2Dataset(cfg,cfg.val_data,'valid') 
    valid_dataloader = DataLoader(valid_dataset,batch_sampler=batch_sampling(cfg.batch_size,len(valid_dataset)))
    tokenizer = valid_dataset.get_tokenizer()




model.resize_token_embeddings(len(tokenizer))
best_prec1 = 0

if cfg.evaluate or cfg.test or cfg.resume:
    torch.cuda.empty_cache()
    model = model.to(device)
    def resume():
        if os.path.isfile(cfg.resume_model):
            print("=> loading checkpoint '{}'".format(cfg.resume_model))
            checkpoint = torch.load(cfg.resume_model)
            cfg.start_epoch = checkpoint['epoch']+1
            global best_prec1
            if 'TL' in cfg.best_filename:
                best_prec1=0
                cfg.epochs = cfg.start_epoch + cfg.epochs
            else:
                best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(cfg.resume_model, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume_model))
    resume()

if cfg.train:
    model = nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count()))).to(device)
    for epoch in range(cfg.start_epoch, cfg.epochs):
        # train for one epoch
        train(train_dataloader, model, optimizer, scheduler, epoch, cfg.epochs)

        # evaluate on validation set
        prec1 = validate(valid_dataloader, model, tokenizer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_model = model
        print(f"Best F1 :{best_prec1}")
        
    save_checkpoint({
        'epoch': epoch,
        'state_dict': best_model.module.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best,filename=cfg.checkpoint_filename,best_filename=cfg.best_filename)



if cfg.evaluate:
    if cfg.train:
        print(f'***On Test Set-{cfg.label_col}***')
        validate(test_dataloader,model,tokenizer)
    else:
        model = nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count()))).to(device)
        validate(valid_dataloader, model, tokenizer)
    # exit()

if cfg.test:
    if cfg.train:
        model = best_model
    if cfg.evaluate:
        test_dataset = CSV2Dataset(cfg,cfg.test_data,'test')
        test_dataloader = DataLoader(test_dataset,batch_sampler=batch_sampling(cfg.batch_size,len(test_dataset),is_test=True))

    tokenizer = test_dataset.get_tokenizer()
    sents, preds = test(test_dataloader,model,tokenizer)


    df = pd.read_csv(cfg.test_data)
    if len(df) != len(preds):
        preds += [''] * (len(df)-len(preds))
    df['decoded_text'] = sents
    df[cfg.test_res_col] = preds
    df.to_csv(cfg.test_res,encoding='utf-8',index=False)
    
