# RapGPT 2.0: GPT-2 based Rap Lyric Generator

RapGPT is a pre-trained & fine-tuned GPT-2 model designed to generate original rap lyrics similar to Eminem's Lyric style based on user prompts. Whether you're writing your next mixtape or experimenting with AI-generated verses, RapGPT has your back.

---

## RapGPT Features

- Prompt-based rap lyric generation
- Adjustable creativity (temperature, top-k, top-p)
- Pre-trained GPT-2 Model (124M) on 17273 songs from 226 artists
- Fine-tuned GPT-2 model trained on Eminmen rap lyrics
- FastAPI backend and React/Next.js frontend with live generation
- Text-to-Speech feature using ElevenLabs API

---
## Installation

Install the required dependencies using pip:


---

## Model Details

- **Base Model**: GPT-2 (774M parameters)
- **Dataset**: ~163,000 cleaned rap lyrics (sourced and filtered from Genius and other sources)
- **Tokenization**: Hugging Face GPT-2 tokenizer with Byte-Pair Encoding (BPE)

### Training Configuration

- **Optimizer**: AdamW
- **Epochs**: *[Your value here]*
- **Batch Size**: *[Your value here]*
- **Loss Function**: CrossEntropyLoss
- **Hardware**: Trained on NVIDIA RTX 4080 and AWS EC2



### Output Format

- Generates raw text with **one bar per line**, mimicking natural rap structure


## 🎤 Example Outputs

**Prompt**: `"Stackin' paper like a CEO"`  
**Output**: `"Stackin' paper like a CEO
Got the hustle, never movin' slow
Flippin' flows like a domino
Bars so sick, they overdose"`

---

## API Usage

### Endpoint

`POST /generate`

---

### Request Body

```json
{
  "prompt": "Feels so empty wihtout me",
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 0.95,
  "max_tokens": 80
}

### Response
{
  "generated_text": "I'm on the grind every day\nStackin' my chips while I pave the way..."
}

## Limitations

- May produce inappropriate outputs  
- Not conditioned on rhythm or beat structure  
- Limited artist-style (Eminem-Only)


---

## Future Improvements

- Add beat/meter alignment for better rhythmic flow  
- Add mutliple artist-styles with fine-tuning



---

## 🙏 Acknowledgments

- 🤗 Hugging Face Transformers  
- 🧠 PyTorch  
- 🎶 Genius.com for dataset inspiration  
- 🧱 OpenAI GPT-2 architecture  
- 💻 React/Next.js frontend community  

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Contact

Questions, feedback, or collab ideas? Reach out: [you@example.com](mailto:you@example.com)

