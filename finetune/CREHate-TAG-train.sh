python finetune.py --config_file ./config_files/BERTweet-CREHate-ALL-tag-train.json | tee -a ./res/BERTweet-CREHate-ALL-tag-train-US.txt
python finetune.py --config_file ./config_files/HateBERT-CREHate-ALL-tag-train.json | tee -a ./res/HateBERT-CREHate-ALL-tag-train-US.txt
python finetune.py --config_file ./config_files/TwHIN-BERT-CREHate-ALL-tag-train.json  | tee -a ./res/TwHIN-BERT-CREHate-ALL-tag-train-US.txt
python finetune.py --config_file ./config_files/Twitter-RoBERTa-CREHate-ALL-tag-train.json  | tee -a ./res/Twitter-RoBERTa-CREHate-ALL-tag-train-US.txt
python finetune.py --config_file ./config_files/ToxDect-RoBERTa-CREHate-ALL-tag-train.json  | tee -a ./res/ToxDect-RoBERTa-CREHate-ALL-tag-train-US.txt
python finetune.py --config_file ./config_files/BERT-CREHate-ALL-tag-train.json  | tee -a ./res/BERT-CREHate-ALL-tag-train-US.txt
python finetune.py --config_file ./config_files/RoBERTa-CREHate-ALL-tag-train.json  | tee -a ./res/RoBERTa-CREHate-ALL-tag-train-US.txt

