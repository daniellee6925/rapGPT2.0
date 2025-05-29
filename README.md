# RapGPT 2.0: GPT-2 based Rap Lyric Generator

RapGPT is a pre-trained & fine-tuned GPT-2 model designed to generate original rap lyrics similar to Eminem's Lyric style based on user prompts. This is an improved version of [rapGPT 1.0](https://github.com/daniellee6925/rapGPT) which is a much smaller bigram model. This project was created for educational purposes, with the goal of learning the full end-to-end lifecycle of large language model (LLM) development—including data collection, pre-training, parameter-efficient fine-tuning (PEFT), model evaluation, and deployment.

If you'd like to try the final product, you can access it here: [www.eminemgpt.com](www.eminemgpt.com])


---

## RapGPT Features

- Prompt-based rap lyric generation
- Adjustable creativity (temperature, top-k, top-p)
- Pre-trained GPT-2 Model (124M) on 17,273 songs (~25M tokens) from 226 artists
- Fine-tuned GPT-2 model trained on Eminmen rap lyrics
- FastAPI backend and React/Next.js frontend with live generation
- Text-to-Speech feature using ElevenLabs API

---
## Installation

Install the required dependencies using pip:
`pip install torch numpy pandas transformers datasets tiktoken`


# Quick Start: Train a GPT-2 model on rap lyrics


## 1. Prepare the Dataset

First, download the dataset 'Lyrics_Data' (~11MB) text file:


`python data/Lyrics_Data`

## 2. Train the GPT-2 model.
If you have access to a GPU, you can start training by running train.py with the appropriate hyperparameters.

`python train.py`

By default, the script uses:
A batch size of 8 and 200 training steps, which corresponds to approximately 4 epochs. You may need to adjust these settings depending on your available GPU memory and dataset size.

Training takes around ~2hrs on RXT 4080.

No GPU?

You can still train the model using a cloud-based GPU instance from providers like AWS EC2, Lambda Labs, or Paperspace.
Training takes around ~3hrs on AWS g4dn.xlarge.

Model will be saved in Models/GPT2_Final

### Model Details

- **Base Model**: GPT-2 (124M parameters)
- **Dataset**: ~17,000 cleaned rap songs (sourced and filtered from Genius.com)
- **Tokenization**: Tiktoken

### Training Configuration

- **Optimizer**: AdamW with cosine decay
- **Epochs**: 4
- **Batch Size**: 8
- **Loss Function**: CrossEntropyLoss
- **Hardware**: Trained on NVIDIA RTX 4080 and AWS EC2 (g4dn.xlarge)


## 3. Generate outputs.
Run generate script after training is complete. 
Modify the prompt (which will be the start of your rap verse).
Modify number of sequences and max tokens

Defaults to: 
1 sequence with 100 tokens

`python generate.py`

### Example Outputs
**Prompt**: "Feels so empty without me"
**Output**: "Feels so empty without me
It feels so empty without me i feel like nothing feels so real
i feel like nothing feels so real so when i see you i know you're mine
so when i see you i know you're mine"
---


## 4. Finetuning.
To fine-tune the model efficiently, you can use LoRA (Low-Rank Adaptation) by running:

`python finetune.py`
You can use lyrics data with an artist of your choice in a .txt file
Modify the data in the """Load and tokenize dataset""" section

LoRA significantly reduces the number of trainable parameters by injecting low-rank adapters into the model's architecture. This method updates only about 1.02% of the total parameters, making training faster and more memory-efficient—especially useful on limited hardware.

---

## 5. Evaluation

To assess the quality and relevance of generated rap lyrics, use a DistilBERT-based classifier model and cosine similarity metrics.

### Evaluation Method

1. **Embedding Generation**  
   Each generated lyric is embedded using a pre-trained DistilBERT model (e.g., `distilbert-base-uncased`) to obtain a semantic vector representation.

2. **Reference Comparison**  
   Generated lyrics are compared against a set of real rap lyric embeddings from the training dataset or manually curated reference lyrics.

3. **Cosine Similarity**  
   Compute the cosine similarity between generated lyrics and reference lyrics:
   Higher scores indicate greater semantic alignment with real-world rap lyric style.

4. **Thresholding / Labeling (Optional)**  
   Optionally, the classifier can be fine-tuned to distinguish between “authentic-style” and “off-style” outputs, assigning a quality label or confidence score to each generation.

### Example Result

| Prompt                          | Cosine Similarity | Comment                   |
|---------------------------------|-------------------|---------------------------|
| `"Chasin' dreams in the rain"` | 0.82              | Strong stylistic match    |
| `"I love data science"`        | 0.48              | Off-topic / unnatural     |
| `"Diamonds dancin' in the sun"`| 0.86              | Excellent stylistic match |


---

## 6. Deployment
rapGPT consists of a **FastAPI backend** and a **React/Next.js frontend**. You can deploy them together or separately depending on your setup (local machine, EC2, Docker, etc.).

Please refer to the repository below

- [backend](https://github.com/daniellee6925/rapGPT_backend])
- [frontend](https://github.com/daniellee6925/rapGPT_frontend])

---
## Limitations

- May produce inappropriate outputs  
- Not conditioned on rhythm or beat structure  
- Limited artist-style (Eminem-Only)

---

## Future Improvements

- Add beat/meter alignment for better rhythmic flow  
- Add mutliple artist-styles with fine-tuning


---

## Acknowledgments

- Hugging Face Transformers  
- PyTorch  
- Genius.com for lyrics dataset 
- OpenAI GPT-2 architecture  
- React/Next.js frontend  

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Contact

Questions, feedback, or collab ideas? Reach out: [daniellee6925@gmail.com](mailto:daniellee6925@gmail.com)

