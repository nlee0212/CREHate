
import os
import re
import csv
import json
import time
import openai
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from easydict import EasyDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, pipeline
import torch




def raw2prediction(x, hn_or_nh):
    if hn_or_nh == 'hn':
        choices = ['Hate', 'Non-hate']
    elif hn_or_nh == 'nh':
        choices = ['Non-hate', 'Hate']
    choice_alphabets = ['a', 'b']
    choice2idx = {'a': 0, 'b': 1}
    
    try:
        raw = x.strip()
    except:
        print(x)
    
    if (raw.startswith('a') and 'a\n' in raw) or 'A is the correct answer' in raw:
        return choices[choice2idx['a']]
    
    if raw.startswith('b') and 'b\n' in raw:
        return choices[choice2idx['b']]
#     "I'd say that's a Hate", "It's not a Hate"
    if 'post is not hate' in raw.lower() or "it's non-hate" in raw.lower() or 'is not hate' in raw.lower() or "it's not hate" in raw.lower() or "it's not a hate" in raw.lower() or "don't think this is hate" in raw.lower() or "is not a hate" in raw.lower() or "would not call it hate" in raw.lower() or "would not consider it as a hate" in raw.lower():
        return 'Non-hate'
    
    if raw.startswith('Hate.') or raw.startswith('Hate Speech') or raw.startswith('Hate\n') or 'it is hate' in raw.lower() or "that's a hate" in raw.lower() or "think it's hate" in raw.lower() or "this is a hate" in raw.lower():
        return 'Hate'
    
    if raw.startswith('Non-hate.') or raw.startswith('Non-hate\n'):
        return 'Non-hate'
    
    if raw.startswith('Comment:'):
        raw = raw.replace('Comment:','')
    
    if 'Classify this text as' in raw:
        raw = re.sub('Classify this text as[\s\S]+','', raw)
    
    if 'answer:' in raw:
        raw = re.sub('[\s\S]+answer:\s','', raw)
        # print(raw)
    
    if 'Answer:' in raw:
        raw = re.sub('[\s\S]+Answer:\s','', raw)
        # print(raw)
        
    if 'can be classified as ' in raw:
        raw = re.sub('[\s\S]+can be classified as ','', raw)
        
    if 'can be categorized as ' in raw:
        raw = re.sub('[\s\S]+can be categorized as ','', raw)
        # print(raw)
        
    if 'is classified as ' in raw:
        raw = re.sub('[\s\S]+is classified as ','', raw) 
    
    raw = raw.replace('hateful','Hate')
    raw = raw.replace('hate speech','Hate')
    raw = raw.replace('Hate speech','Hate')
    raw = raw.replace('###','')
    
    
    if 'can be considered hateful' in raw:
        return 'Hate'

    if 'does not contain hate' in raw:
        return 'Non-hate'
    
    try:
        raw = re.search('\*\*\s*(?P<raw>.*)\s*\*\*', raw).groupdict()['raw']
    except:
        pass
    
    if 'answer is' in raw:
        regex = 'answer is\s*(?P<answer>[^\.\n<]*)'
    else:
        regex = 'Answer\s*:\s*(?P<answer>[^\.\n<]*)'

    try:
        prediction = re.search(regex, raw).groupdict()['answer']
    except:
        prediction = raw
    
    try:
        if raw.strip()[0] == '(':
            regex = '(Option|option|[\*\s]*)\s*(?P<answer>[^\.\n\*<]*)'
        else:
            regex = '(Option|option|[\*\s]*)\s*(?P<answer>[^\.\n\*(<]*)'
    except:
        if raw.strip() == '(':
            regex = '(Option|option|[\*\s]*)\s*(?P<answer>[^\.\n\*<]*)'
        else:
            regex = '(Option|option|[\*\s]*)\s*(?P<answer>[^\.\n\*(<]*)'
    try:
        prediction = re.search(regex, prediction).groupdict()['answer']
    except:
        prediction = re.search('\s*(?P<answer>[^\.\n\*(<]*)', prediction).groupdict()['answer']
    
    prediction = re.sub('[^\S ]', '', prediction.strip(" \n\t'\"()"))

    if len(prediction) == 0:
        prediction = raw
    if len(prediction) == 0:
        return ''
    prediction_upper = prediction.lower()

    if (prediction_upper[0] in choice_alphabets):
        try:
            choice = re.search('[:)\-]\s*(?P<choice>[^(]*)', prediction_upper).groupdict()['choice']
        except:
            if prediction_upper.count(choice_alphabets[0]) + prediction_upper.count(choice_alphabets[1]) == 1:
                prediction = prediction_upper[0]
            
            if prediction in choice_alphabets:
                prediction = choices[choice2idx[prediction]]
            
            if prediction.startswith('non'):
                prediction = 'Non-hate'
            if prediction in choice_alphabets:
                prediction = choices[choice2idx[prediction]]
            if prediction in ['hate','H','h','H for Hate','a" Hate','Yes, a: Hate','This post would be considered as a Hate',
                             '[a] Hate','b, Hate','b Hate','h: Hate','B  Hate','b  Hate','Hate Speech','b  Hate','a" Hate',
                             'This text is hate','HATE',"I think it's hate","I think it's a hate post",'\\strong{Hate}',
                             'I think this is a hate post','hate"?']:
                prediction = 'Hate'
            if prediction in ['Non-Hate Speech','The post provided does not contain Hate',
                             'Non-hate','I: Non-hate','Not-hate','n','N','No, b: Non-hate',
                             'Not enough context to determine if it is Hate or not, "b: Non-hate','Non-hate a',
                              'Non-hate Speech','Non-Hate','It is not hate',"I think it's non-hate","I think it's a non-hate post",
                              "It's not hate",'Non','This is non-hate','Non-'
                             ]:
                prediction = 'Non-hate'
            if prediction == '1':
                prediction = choices[0]
                
            return prediction
        
        if choices[choice2idx[prediction_upper[0]]].lower() == choice.lower():
            prediction = prediction_upper[0]
        else:
            prediction = prediction_upper[0]
    if prediction.startswith('non'):
        prediction = 'Non-hate'
    if prediction in choice_alphabets:
        prediction = choices[choice2idx[prediction]]
    if prediction in ['hate','H','h','H for Hate','a" Hate','Yes, a: Hate','This post would be considered as a Hate',
                     '[a] Hate','b, Hate','b Hate','h: Hate','B  Hate','b  Hate','Hate Speech','b  Hate','a" Hate',
                     'This text is hate','HATE',"I think it's hate","I think it's a hate post",'\\strong{Hate}',
                     'I think this is a hate post','hate"?']:
        prediction = 'Hate'
    if prediction in ['Non-Hate Speech','The post provided does not contain Hate',
                     'Non-hate','I: Non-hate','Not-hate','n','N','No, b: Non-hate',
                     'Not enough context to determine if it is Hate or not, "b: Non-hate','Non-hate a',
                      'Non-hate Speech','Non-Hate','It is not hate',"I think it's non-hate","I think it's a non-hate post",
                      "It's not hate",'Non','This is non-hate','Non-'
                     ]:
        prediction = 'Non-hate'
    if prediction == '1':
        prediction = choices[0]
        
    
        
    return prediction

def prediction_2_label(prediction):
    prediction_lower = prediction.lower()
    if prediction_lower in ['a', ab2label['a'].lower(), ab2label['a'].lower().replace('-', ''), f"{ab2label['a'].lower()} (a)", f"a ({ab2label['a'].lower()})", f"a: {ab2label['a'].lower()}", f"a. {ab2label['a'].lower()}"]:
        return ab2label['a']
    if prediction_lower in ['b', ab2label['b'].lower(), ab2label['b'].lower().replace('-', ''), f"{ab2label['b'].lower()} (b)", f"b ({ab2label['b'].lower()})", f"b: {ab2label['b'].lower()}", f"b. {ab2label['b'].lower()}"]:
        return ab2label['b']
    
    if prediction not in ['Hate', 'Non-hate']:
        print(prediction)
        
    return prediction

def make_prompt(instance,ab2label,persona=False,country=None,simple=False,definition=False,prompt_num=None):
    if prompt_num == None:
        prompt_num = 0
    
    prompts = [
        'Answer if this post is hate or not with a single alphabet letter among given answer choices a and b.',
        "Given a POST from online, answer if it is 'Hate Speech' or 'Non-Hate Speech'. Answer with only one single alphabet between 'a' and 'b', with no other description.",
        "Would you consider this post as a hate speech? Give an answer in a single alphabet only.",
        "Do you think this is a hate speech? Answer with only one single alphabet between 'a' and 'b', with no other description.",
        "Consider yourself as a very typical person. Would you consider this post as a hate speech? Give an answer in a single alphabet only."
    ]
    
    prompts_w_persona = [
        f"Answer if this post is hate or not in {country} with a single alphabet letter among given answer choices a and b.",
        f"Given a POST from online, answer if it is 'Hate Speech' or 'Non-Hate Speech' in {country}. Answer with only one single alphabet between 'a' and 'b', with no other description.",
        f"As a person raised up in {country}, would you consider this post as a hate speech? Give an answer in a single alphabet only.",
        f"You are from {country}. Do you think this is a hate speech? Answer with only one single alphabet between 'a' and 'b', with no other description.",
        f"Consider yourself as a very typical person in {country}. Would you consider this post as a hate speech? Give an answer in a single alphabet only."
    ]
    
    p = ''
    
    if simple:
        p = f'Classify this text as {ab2label["a"]} or {ab2label["b"]}. {instance}. Answer:'
        return p
    
    if definition:
        p = f"Definition of Hate Speech:\n\nHate speech refers to offensive discourse targeting a group or an individual based on inherent characteristics such as race, religion, sexual orientation, gender, or any other factors that may threaten social peace.\n\n" 
    
    if persona:
        p += prompts_w_persona[prompt_num]
    else:
        p += prompts[prompt_num]
    
    p+='\n\n'
        
    p += f'POST: {instance}\n'
    p += f'a: {ab2label["a"]}\n'
    p += f'b: {ab2label["b"]}\n'
    p += 'answer:'

    return p


openai.organization = OPENAI_ORGANIZATION_KEY
openai.api_key = OPENAI_API_KEY


def check_gpt_input_list(history):
    check = True
    for i, u in enumerate(history):
        if not isinstance(u, dict):
            check = False
            break
            
        if not u.get("role") or not u.get("content"):
            check = False
            break
        
    return check


def get_gpt_response(
    text,
    model_name,
    temperature=1.0,
    top_p=1.0,
    max_tokens=128,
    greedy=False,
    num_sequence=1,
    max_try=60,
    dialogue_history=None
):
    # assert model_name in GPT_MODEL

    if (model_name.startswith("gpt-3.5-turbo") and 'instruct' not in model_name) or model_name.startswith("gpt-4"):
        if dialogue_history:
            if not check_gpt_input_list(dialogue_history):
                raise Exception("Input format is not compatible with chatgpt api! Please see https://platform.openai.com/docs/api-reference/chat")
            messages = dialogue_history
        else:
            messages = []
        
        messages.append({'role': 'user', 'content': text})

        prompt = {
            "model": model_name,
            "messages": messages,
            "temperature": 0. if greedy else temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": num_sequence
        }

    else:    
        prompt = {
            "model": model_name,
            "prompt": text,
            "temperature": 0. if greedy else temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": num_sequence
        }
    
    n_try = 0
    while True:
        if n_try == max_try:
            outputs = ["something wrong"]
            break
        
        try:
            if (model_name.startswith("gpt-3.5-turbo") and 'instruct' not in model_name) or model_name.startswith("gpt-4"):
                time.sleep(0.5)
                res = openai.ChatCompletion.create(**prompt)
                outputs = [o['message']['content'].strip("\n ") for o in res['choices']]
            else:
                res = openai.Completion.create(**prompt)
                outputs = [o['text'].strip("\n ") for o in res['choices']]
            break
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except:
            print("Exception: Sleep for 10 sec")
            time.sleep(10)
            n_try += 1
            continue
        
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs

sbic_data = pd.read_csv(PATH_TO_SBIC_DATA,index_col=False)
additional_data = pd.read_csv(PATH_TO_CP_DATA,index_col=False)


models_to_eval = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-70b-hf',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-instruct',
    'gpt-3.5-turbo-1106',
    'gpt-4-1106-preview',
    'google/flan-t5-small','google/flan-t5-base','google/flan-t5-large',
    'google/flan-t5-xl',
    'google/flan-t5-xxl',
    'facebook/opt-iml-30b'
    "microsoft/Orca-2-13b",
    'microsoft/Orca-2-7b',
    ]



output_dir = Path(PATH_TO_OUTPUT_DIRECTORY) 

countries = ['Australia','United States','United Kingdom','South Africa','Singapore']

for model_name in models_to_eval:

    if 'flan-t5' in model_name: 
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", 
                                                            resume_download=True,
                                                            cache_dir=f'.cache/{model_name}')
    
    elif 'llama' in model_name or 'LLaMa' in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False,token=HUGGINGFACE_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", 
                                                            torch_dtype=torch.float16,
                                                            resume_download=True,
                                                            cache_dir=f'.cache/{model_name}',use_auth_token=HUGGINGFACE_TOKEN)
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", 
                                                            resume_download=True,
                                                            cache_dir=f'.cache/{model_name}') 
    
    def model_infer(persona=False,country=None,simple=False,definition=True,prompt_num=None):
        with open(output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'post', 'US', 'AU', 'GB', 'ZA', 'SG', 'prediction', 'raw'])
        done_id = pd.read_csv(output_path, encoding='utf-8')['id'].to_list()

        total_num = len(data)

        hit_us, hit_uk, hit_au, hit_sa, hit_sg = 0, 0, 0, 0, 0
        evaluated_num = 0
        ooc = 0
        
        tqdm_label = f'{model_name}-{prompt_num}'
        if persona:
            tqdm_label += f'-{country}'
        
        for idx, instance in tqdm(data.iterrows(), total=total_num, desc=tqdm_label):

            if instance['ID'] in done_id:
                continue

            evaluated_num += 1

            
            if persona:
                prompt = make_prompt(instance[post_col],ab2label,persona=persona,country=country,definition=definition,prompt_num=prompt_num)
            elif simple:
                prompt = make_prompt(instance[post_col],ab2label,simple=simple)
            else:
                prompt = make_prompt(instance[post_col],ab2label,definition=definition,prompt_num=prompt_num)
            print(prompt)



            
            if model_name.startswith('gpt'):
                result = get_gpt_response(prompt,model_name)
                raw = result
                prediction = raw2prediction(result,sequence)
                print(raw)
                print(prediction)
                label = prediction_2_label(prediction)
            
            else:
                input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

                outputs = model.generate(**input_ids,max_new_tokens=30)
                result = tokenizer.decode(outputs[0],skip_special_tokens=True)

                raw = result.replace(prompt, '')
                prediction = raw2prediction(raw,sequence)
                print('raw:')
                print(raw)
                print('pred:')
                print(prediction)
                label = prediction_2_label(prediction)
            
            if label not in ab2label.values():
                ooc += 1
                print('# ooc =', ooc)

            if label == num2label[int(float(instance['United_States_Hate']))]:
                hit_us += 1
            if label == num2label[int(float(instance['United_Kingdom_Hate']))]:
                hit_uk += 1
            if label == num2label[int(float(instance['Australia_Hate']))]:
                hit_au += 1
            if label == num2label[int(float(instance['South_Africa_Hate']))]:
                hit_sa += 1
            if label == num2label[int(float(instance['Singapore_Hate']))]:
                hit_sg += 1

            open_trial = 0
            while True:
                if open_trial > 10:
                    raise Exception("something wrong")

                try:
                    with open(output_path, "a", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            instance['ID'],
                            instance[post_col],
                            num2label[int(float(instance['United_States_Hate']))],
                            num2label[int(float(instance['Australia_Hate']))],
                            num2label[int(float(instance['United_Kingdom_Hate']))],
                            num2label[int(float(instance['South_Africa_Hate']))],
                            num2label[int(float(instance['Singapore_Hate']))],
                            label,
                            raw])
                    break
                except:
                    print("open failed")
                    continue
            print(f"[{model_name}]\tUS: {hit_us / evaluated_num:.4f}\tAU: {hit_au / evaluated_num:.4f}\tUK: {hit_uk / evaluated_num:.4f}\tSA: {hit_sa / evaluated_num:.4f}\tSG: {hit_sg / evaluated_num:.4f}")

    
   
    num2label = ['Non-hate', 'Hate']
    post_col = 'Text' 
    
    # TO CHANGE
    # Both False: Use the original prompt without the country information
    # Persona True: Use the prompt with the country information
    # Simple True: Only use the simple prompt 'Classify this text as {ab2label["a"]} or {ab2label["b"]}. {instance}. Answer:'
    persona = False 
    simple = False  

    if simple:
        ab2label = {'a': 'Hate', 'b': 'Non-hate'}
        sequence = 'hn'
        label2ab = {v:k for k,v in ab2label.items()}
        id2ab = {1:label2ab['Hate'],0:label2ab['Non-hate']} 
        
        data = additional_data
        
        output_path = output_dir / f"{model_name.replace('/','-')}_simpleprompt_additional_predictions_{ab2label['a']}_{ab2label['b']}.csv"
        model_infer(simple=simple)
        
        data = sbic_data
        
        output_path = output_dir / f"{model_name.replace('/','-')}_simpleprompt_predictions_{ab2label['a']}_{ab2label['b']}.csv"
        model_infer(simple=simple)
        
        ab2label = {'a': 'Non-hate', 'b': 'Hate'}
        sequence = 'nh'
        label2ab = {v:k for k,v in ab2label.items()}
        id2ab = {1:label2ab['Hate'],0:label2ab['Non-hate']}
        
        data = additional_data
        
        output_path = output_dir / f"{model_name.replace('/','-')}_simpleprompt_additional_predictions_{ab2label['a']}_{ab2label['b']}.csv"
        model_infer(simple=simple)
        
        data = sbic_data
        
        output_path = output_dir / f"{model_name.replace('/','-')}_simpleprompt_predictions_{ab2label['a']}_{ab2label['b']}.csv"
        model_infer(simple=simple) 
    
    elif persona:
        for i in range(5): #PROMPTS
            for country in countries:
                
                ab2label = {'a': 'Hate', 'b': 'Non-hate'}
                sequence = 'hn'
                label2ab = {v:k for k,v in ab2label.items()}
                id2ab = {1:label2ab['Hate'],0:label2ab['Non-hate']} 
                
                data = additional_data
                output_path = output_dir / f"{model_name.replace('/','-')}_{country.replace(' ','_')}_add_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv" 
                model_infer(definition=True,prompt_num=i,persona=persona,country=country)
                
                data = sbic_data
                output_path = output_dir / f"{model_name.replace('/','-')}_{country.replace(' ','_')}_sbic_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv" 
                model_infer(definition=True,prompt_num=i,persona=persona,country=country)
                
                ab2label = {'a': 'Non-hate', 'b': 'Hate'}
                sequence = 'nh'
                label2ab = {v:k for k,v in ab2label.items()}
                id2ab = {1:label2ab['Hate'],0:label2ab['Non-hate']} 
                
                data = additional_data
                output_path = output_dir / f"{model_name.replace('/','-')}_{country.replace(' ','_')}_add_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv" 
                model_infer(definition=True,prompt_num=i,persona=persona,country=country)
                
                data = sbic_data
                output_path = output_dir / f"{model_name.replace('/','-')}_{country.replace(' ','_')}_sbic_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv" 
                model_infer(definition=True,prompt_num=i,persona=persona,country=country)
            
    
    else:
        
        for i in range(5): #PROMPTS

              
            ab2label = {'a': 'Hate', 'b': 'Non-hate'}
            sequence = 'hn'
            label2ab = {v:k for k,v in ab2label.items()}
            id2ab = {1:label2ab['Hate'],0:label2ab['Non-hate']} 
            
            data = additional_data
            output_path = output_dir / f"{model_name.replace('/','-')}_add_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv" 
            model_infer(definition=True,prompt_num=i)
            
            data = sbic_data
            output_path = output_dir / f"{model_name.replace('/','-')}_sbic_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv" 
            model_infer(definition=True,prompt_num=i)
            
            ab2label = {'a': 'Non-hate', 'b': 'Hate'}
            sequence = 'nh'
            label2ab = {v:k for k,v in ab2label.items()}
            id2ab = {1:label2ab['Hate'],0:label2ab['Non-hate']} 
            
            data = additional_data
            output_path = output_dir / f"{model_name.replace('/','-')}_add_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv" 
            model_infer(definition=True,prompt_num=i)
            
            data = sbic_data
            output_path = output_dir / f"{model_name.replace('/','-')}_sbic_prompt_{i}_w_def_{ab2label['a']}_{ab2label['b']}.csv" 
            model_infer(definition=True,prompt_num=i)
            
            


