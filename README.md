# CREHate

Repository for the CREHate dataset, presented in the paper "Exploring Cross-cultural Differences in English Hate Speech Annotations: From Dataset Construction to Analysis".

## About CREHate (Paper Abstract)
Most NLP datasets neglect the cultural diversity among language speakers, resulting in a critical shortcoming in hate speech detection and other culturally sensitive tasks.
To address this, we introduce **CREHate**, a **CR**oss-cultural **E**nglish **Hate** speech dataset.
To construct CREHate, we follow a two-step procedure: 1) cultural post collection and 2) cross-cultural annotation.
We sample posts from the SBIC dataset, which predominantly represents North America, and collect posts from four geographically diverse English-speaking countries (Australia, United Kingdom, Singapore, and South Africa) using culturally hateful keywords we retrieve from our survey.
Annotations are collected from the four countries plus the US to establish representative labels for each country.
Our analysis highlights statistically significant disparities across countries in hate speech annotations.
Only 56.2\% of the posts in CREHate achieve consensus among all countries, with the highest pairwise label difference rate of 26\%.
Qualitative analysis shows that label disagreement occurs mostly due to different interpretations of sarcasm and the personal bias of annotators on divisive topics.
Lastly, we evaluate large language models (LLMs) on CREHate under a zero-shot setting and show that some models tend to show higher label similarities with Anglosphere countries.

## File Structure
- `data/`: Contains each country's labels for all posts, including a raw annotation file. Separate files for CC-SBIC and CP posts only are also included.
- `finetune/`: Contains codes and training data splits for finetuning BERT-variants (Appendix F)
- `llm_inference.py`: Codes used for LLM inference. OpenAI key is needed for inference in GPT models.
