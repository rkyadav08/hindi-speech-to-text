# 🎙️ Hindi Speech-to-Text — Fine-tuned Whisper-small

Fine-tuned OpenAI's Whisper-small on ~10 hours of Hindi conversational audio, achieving a **29.18 percentage point WER reduction** on the FLEURS Hindi benchmark.

🔗 **Live Demo:** [hindi-speech-to-text on HuggingFace Spaces](https://huggingface.co/spaces/rkyadav08/hindi-speech-to-text)

---

## 📊 Results

| Model | WER (%) |
|-------|---------|
| Whisper-small (pretrained) | 68.35 |
| Whisper-small (fine-tuned) | **39.17** |
| **Improvement** | **+29.18 pp (42.7% relative)** |

---

## 🏗️ Architecture

```
Audio (16kHz) → Log-Mel Spectrogram → Whisper Encoder (12 layers)
→ Whisper Decoder (autoregressive) → Hindi Text
```

**Base Model:** `openai/whisper-small` (244M parameters)  
**Evaluation Set:** FLEURS Hindi (hi_in) — 418 utterances

---

## ⚙️ Training

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-5 |
| Batch Size | 16 |
| Max Steps | 2,000 |
| Warmup Steps | 500 |
| Precision | FP16 |
| GPU | Google Colab T4 |

---

## 🚀 Live Demo

Deployed on **HuggingFace Spaces** → [Try it here](https://huggingface.co/spaces/rkyadav08/hindi-speech-to-text)

- Upload audio or record from microphone
- Dictation mode for continuous transcription
- Supports WAV, MP3, FLAC, OGG, M4A

---

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Gradio web application |
| `model.safetensors` | Fine-tuned model weights (967 MB) |
| `config.json` | Model configuration |
| `tokenizer.json` | Whisper tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `vocab.json` | Vocabulary file |
| `merges.txt` | BPE merges |
| `normalizer.json` | Text normalizer |
| `added_tokens.json` | Additional tokens |
| `special_tokens_map.json` | Special token mappings |
| `preprocessor_config.json` | Audio preprocessor config |
| `generation_config.json` | Generation parameters |
| `requirements.txt` | Python dependencies |

---

## 💻 Run Locally

```bash
pip install torch torchaudio transformers librosa soundfile gradio

# Run the web app
python app.py
# Opens at http://localhost:7860
```

---

## 🛠️ Tech Stack

**PyTorch** · **HuggingFace Transformers** · **Whisper** · **Gradio** · **Google Colab**

---

## 📝 License

MIT License
