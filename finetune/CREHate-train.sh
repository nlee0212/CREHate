# US
python finetune.py --config_file ./config_files/BERTweet-CREHate-ALL-train.json | tee -a ./res/BERTweet-CREHate-ALL-train-US.txt
python finetune.py --config_file ./config_files/HateBERT-CREHate-ALL-train.json | tee -a ./res/HateBERT-CREHate-ALL-train-US.txt
python finetune.py --config_file ./config_files/TwHIN-BERT-CREHate-ALL-train.json  | tee -a ./res/TwHIN-BERT-CREHate-ALL-train-US.txt
python finetune.py --config_file ./config_files/Twitter-RoBERTa-CREHate-ALL-train.json  | tee -a ./res/Twitter-RoBERTa-CREHate-ALL-train-US.txt
python finetune.py --config_file ./config_files/ToxDect-RoBERTa-CREHate-ALL-train.json  | tee -a ./res/ToxDect-RoBERTa-CREHate-ALL-train-US.txt
python finetune.py --config_file ./config_files/BERT-CREHate-ALL-train.json  | tee -a ./res/BERT-CREHate-ALL-train-US.txt
python finetune.py --config_file ./config_files/RoBERTa-CREHate-ALL-train.json  | tee -a ./res/RoBERTa-CREHate-ALL-train-US.txt

# AU
python finetune.py --config_file ./config_files/BERTweet-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[AU]BERTweet-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[AU]BERTweet-base_CREHate_best.pth.tar --label_col Australia_Hate --test_res_col BERTweet-CREHate-AU  | tee -a ./res/BERTweet-CREHate-ALL-train-AU.txt
python finetune.py --config_file ./config_files/HateBERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[AU]HateBERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[AU]HateBERT-base_CREHate_best.pth.tar --label_col Australia_Hate --test_res_col HateBERT-CREHate-AU | tee -a ./res/HateBERT-CREHate-ALL-train-AU.txt
python finetune.py --config_file ./config_files/TwHIN-BERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[AU]TwHIN-BERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[AU]TwHIN-BERT-base_CREHate_best.pth.tar --label_col Australia_Hate --test_res_col TwHIN-BERT-CREHate-AU | tee -a ./res/TwHIN-BERT-CREHate-ALL-train-AU.txt
python finetune.py --config_file ./config_files/Twitter-RoBERTa-CREHate-ALL-train.json  --checkpoint_filename ./checkpoints/[AU]Twitter-RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[AU]Twitter-RoBERTa-base_CREHate_best.pth.tar --label_col Australia_Hate --test_res_col Twitter-RoBERTa-CREHate-AU | tee -a ./res/Twitter-RoBERTa-CREHate-ALL-train-AU.txt
python finetune.py --config_file ./config_files/ToxDect-RoBERTa-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[AU]ToxDect-RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[AU]ToxDect-RoBERTa-base_CREHate_best.pth.tar --label_col Australia_Hate --test_res_col ToxDect-RoBERTa-CREHate-AU | tee -a ./res/ToxDect-RoBERTa-CREHate-ALL-train-AU.txt
python finetune.py --config_file ./config_files/BERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[AU]BERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[AU]BERT-base_CREHate_best.pth.tar --label_col Australia_Hate --test_res_col BERT-CREHate-AU | tee -a ./res/BERT-CREHate-ALL-train-AU.txt
python finetune.py --config_file ./config_files/RoBERTa-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[AU]RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[AU]RoBERTa-base_CREHate_best.pth.tar --label_col Australia_Hate --test_res_col RoBERTa-CREHate-AU | tee -a ./res/RoBERTa-CREHate-ALL-train-AU.txt

# # # UK
python finetune.py --config_file ./config_files/BERTweet-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[UK]BERTweet-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[UK]BERTweet-base_CREHate_best.pth.tar --label_col United_Kingdom_Hate --test_res_col BERTweet-CREHate-UK  | tee -a ./res/BERTweet-CREHate-ALL-train-UK.txt
python finetune.py --config_file ./config_files/HateBERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[UK]HateBERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[UK]HateBERT-base_CREHate_best.pth.tar --label_col United_Kingdom_Hate --test_res_col HateBERT-CREHate-UK | tee -a ./res/HateBERT-CREHate-ALL-train-UK.txt
python finetune.py --config_file ./config_files/TwHIN-BERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[UK]TwHIN-BERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[UK]TwHIN-BERT-base_CREHate_best.pth.tar --label_col United_Kingdom_Hate --test_res_col TwHIN-BERT-CREHate-UK | tee -a ./res/TwHIN-BERT-CREHate-ALL-train-UK.txt
python finetune.py --config_file ./config_files/Twitter-RoBERTa-CREHate-ALL-train.json  --checkpoint_filename ./checkpoints/[UK]Twitter-RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[UK]Twitter-RoBERTa-base_CREHate_best.pth.tar --label_col United_Kingdom_Hate --test_res_col Twitter-RoBERTa-CREHate-UK | tee -a ./res/Twitter-RoBERTa-CREHate-ALL-train-UK.txt
python finetune.py --config_file ./config_files/ToxDect-RoBERTa-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[UK]ToxDect-RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[UK]ToxDect-RoBERTa-base_CREHate_best.pth.tar --label_col United_Kingdom_Hate --test_res_col ToxDect-RoBERTa-CREHate-UK  | tee -a ./res/ToxDect-RoBERTa-CREHate-ALL-train-UK.txt
python finetune.py --config_file ./config_files/BERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[UK]BERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[UK]BERT-base_CREHate_best.pth.tar --label_col United_Kingdom_Hate --test_res_col BERT-CREHate-UK | tee -a ./res/BERT-CREHate-ALL-train-UK.txt
python finetune.py --config_file ./config_files/RoBERTa-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[UK]RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[UK]RoBERTa-base_CREHate_best.pth.tar --label_col United_Kingdom_Hate --test_res_col RoBERTa-CREHate-UK | tee -a ./res/RoBERTa-CREHate-ALL-train-UK.txt

# # # ZA
python finetune.py --config_file ./config_files/BERTweet-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[ZA]BERTweet-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[ZA]BERTweet-base_CREHate_best.pth.tar --label_col South_Africa_Hate --test_res_col BERTweet-CREHate-ZA  | tee -a ./res/BERTweet-CREHate-ALL-train-ZA.txt
python finetune.py --config_file ./config_files/HateBERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[ZA]HateBERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[ZA]HateBERT-base_CREHate_best.pth.tar --label_col South_Africa_Hate --test_res_col HateBERT-CREHate-ZA | tee -a ./res/HateBERT-CREHate-ALL-train-ZA.txt
python finetune.py --config_file ./config_files/TwHIN-BERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[ZA]TwHIN-BERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[ZA]TwHIN-BERT-base_CREHate_best.pth.tar --label_col South_Africa_Hate --test_res_col TwHIN-BERT-CREHate-ZA | tee -a ./res/TwHIN-BERT-CREHate-ALL-train-ZA.txt
python finetune.py --config_file ./config_files/Twitter-RoBERTa-CREHate-ALL-train.json  --checkpoint_filename ./checkpoints/[ZA]Twitter-RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[ZA]Twitter-RoBERTa-base_CREHate_best.pth.tar --label_col South_Africa_Hate --test_res_col Twitter-RoBERTa-CREHate-ZA | tee -a ./res/Twitter-RoBERTa-CREHate-ALL-train-ZA.txt
python finetune.py --config_file ./config_files/ToxDect-RoBERTa-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[ZA]ToxDect-RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[ZA]ToxDect-RoBERTa-base_CREHate_best.pth.tar --label_col South_Africa_Hate --test_res_col ToxDect-RoBERTa-CREHate-ZA  | tee -a ./res/ToxDect-RoBERTa-CREHate-ALL-train-ZA.txt
python finetune.py --config_file ./config_files/BERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[ZA]BERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[ZA]BERT-base_CREHate_best.pth.tar --label_col South_Africa_Hate --test_res_col BERT-CREHate-ZA | tee -a ./res/BERT-CREHate-ALL-train-ZA.txt
python finetune.py --config_file ./config_files/RoBERTa-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[ZA]RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[ZA]RoBERTa-base_CREHate_best.pth.tar --label_col South_Africa_Hate --test_res_col RoBERTa-CREHate-ZA | tee -a ./res/RoBERTa-CREHate-ALL-train-ZA.txt

# # # SG
python finetune.py --config_file ./config_files/BERTweet-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[SG]BERTweet-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[SG]BERTweet-base_CREHate_best.pth.tar --label_col Singapore_Hate --test_res_col BERTweet-CREHate-SG  | tee -a ./res/BERTweet-CREHate-ALL-train-SG.txt
python finetune.py --config_file ./config_files/HateBERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[SG]HateBERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[SG]HateBERT-base_CREHate_best.pth.tar --label_col Singapore_Hate --test_res_col HateBERT-CREHate-SG | tee -a ./res/HateBERT-CREHate-ALL-train-SG.txt
python finetune.py --config_file ./config_files/TwHIN-BERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[SG]TwHIN-BERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[SG]TwHIN-BERT-base_CREHate_best.pth.tar --label_col Singapore_Hate --test_res_col TwHIN-BERT-CREHate-SG | tee -a ./res/TwHIN-BERT-CREHate-ALL-train-SG.txt
python finetune.py --config_file ./config_files/Twitter-RoBERTa-CREHate-ALL-train.json  --checkpoint_filename ./checkpoints/[SG]Twitter-RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[SG]Twitter-RoBERTa-base_CREHate_best.pth.tar --label_col Singapore_Hate --test_res_col Twitter-RoBERTa-CREHate-SG | tee -a ./res/Twitter-RoBERTa-CREHate-ALL-train-SG.txt
python finetune.py --config_file ./config_files/ToxDect-RoBERTa-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[SG]ToxDect-RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[SG]ToxDect-RoBERTa-base_CREHate_best.pth.tar --label_col Singapore_Hate --test_res_col ToxDect-RoBERTa-CREHate-SG  | tee -a ./res/ToxDect-RoBERTa-CREHate-ALL-train-SG.txt
python finetune.py --config_file ./config_files/BERT-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[SG]BERT-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[SG]BERT-base_CREHate_best.pth.tar --label_col Singapore_Hate --test_res_col BERT-CREHate-SG | tee -a ./res/BERT-CREHate-ALL-train-SG.txt
python finetune.py --config_file ./config_files/RoBERTa-CREHate-ALL-train.json --checkpoint_filename ./checkpoints/[SG]RoBERTa-base_CREHate_checkpoint.pth.tar --best_filename ./checkpoints/[SG]RoBERTa-base_CREHate_best.pth.tar --label_col Singapore_Hate --test_res_col RoBERTa-CREHate-SG | tee -a ./res/RoBERTa-CREHate-ALL-train-SG.txt

