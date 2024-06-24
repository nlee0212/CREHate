# CREHate

Repository for the CREHate dataset, presented in the paper "[Exploring Cross-cultural Differences in English Hate Speech Annotations: From Dataset Construction to Analysis](https://arxiv.org/abs/2308.16705)". (NAACL 2024, Best Resource Awards)

🤗 Our dataset is uploaded on [HuggingFace](https://huggingface.co/datasets/nayeon212/CREHate)!

## About CREHate (Paper Abstract)

<img src="https://github.com/nlee0212/CREHate/blob/main/CREHate_Dataset_Construction.png" width="400">

Most hate speech datasets neglect the cultural diversity within a single language, resulting in a critical shortcoming in hate speech detection. 
To address this, we introduce **CREHate**, a **CR**oss-cultural **E**nglish **Hate** speech dataset.
To construct CREHate, we follow a two-step procedure: 1) cultural post collection and 2) cross-cultural annotation.
We sample posts from the SBIC dataset, which predominantly represents North America, and collect posts from four geographically diverse English-speaking countries (Australia, United Kingdom, Singapore, and South Africa) using culturally hateful keywords we retrieve from our survey.
Annotations are collected from the four countries plus the United States to establish representative labels for each country.
Our analysis highlights statistically significant disparities across countries in hate speech annotations.
Only 56.2% of the posts in CREHate achieve consensus among all countries, with the highest pairwise label difference rate of 26%.
Qualitative analysis shows that label disagreement occurs mostly due to different interpretations of sarcasm and the personal bias of annotators on divisive topics.
Lastly, we evaluate large language models (LLMs) under a zero-shot setting and show that current LLMs tend to show higher accuracies on Anglosphere country labels in CREHate.

## Dataset Statistics
<div id="tab:3_1_stats">

| **Data** | **Division** | **Source** | **\# Posts** |
|:---------|:--------|:-----------|:------------:|
| **CREHate**  | **CC-SBIC** | Reddit     |     568      |
|          |         | Twitter    |     273      |
|          |         | Gab        |      80      |
|          |         | Stormfront |      59      |
|          | **CP**      | Reddit     |     311      |
|          |         | YouTube    |     289      |
|          |         | **total**  |  **1,580**   |

Data statistics and sources of CREHate. CC-SBIC refers to cross-culturally
re-annotated SBIC posts. CP refers to additionally collected cultural
posts from four countries (AU, GB, SG, and ZA), which are also
cross-culturally annotated.

</div>

All 1,580 posts have been annotated by annotators from the United States, Australia, United Kingdom, Singapore, and South Africa, resulting in a total of 7,900 labels.


## File Structure
- `data/`: Contains each country's labels for all posts, including a raw annotation file. Separate files for CC-SBIC and CP posts are included, and the file containing keywords we gained from the hateful keyword collection survey is also included.
- `llm_inference.py`: Codes used for LLM inference. OpenAI key is needed for inference in GPT models. By changing L468, you can test on prompts with country personas.
  
  ```shell
  $ python llm_inference.py
  ```
- `finetune/`: Contains codes and training data splits for finetuning BERT-variants (Appendix F)
