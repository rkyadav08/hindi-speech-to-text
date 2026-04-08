# 🎙️ Hindi Speech Recognition — Fine-tuning Whisper-small

Fine-tuned OpenAI's Whisper-small model on ~10 hours of Hindi conversational audio, achieving a **29.18 percentage point WER reduction** on the FLEURS Hindi benchmark. Includes post-processing pipelines, spelling validation at scale, and a live deployed web app.

🔗 **Live Demo:** [hindi-speech-to-text on HuggingFace Spaces](https://huggingface.co/spaces/rkyadav08/hindi-speech-to-text)

---

## 📊 Results

| Model | WER (%) |
|-------|---------|
| Whisper-small (pretrained) | 68.35 |
| Whisper-small (fine-tuned) | **39.17** |
| **Improvement** | **+29.18 pp (42.7% relative)** |

### Training Progress

| Step | Training Loss | Validation Loss | WER (%) |
|------|--------------|----------------|---------|
| 100 | 0.6985 | 0.7282 | 79.47 |
| 200 | 0.4846 | 0.5582 | 63.69 |
| 300 | 0.3550 | 0.4758 | 57.13 |
| 400 | 0.3209 | 0.4168 | 55.47 |

---

## 🏗️ Architecture

```
Audio (16kHz WAV) → Log-Mel Spectrogram → Whisper Encoder (12 layers)
→ Whisper Decoder (autoregressive) → Hindi Text → Post-Processing → Clean Output
```

- **Base Model:** `openai/whisper-small` (244M parameters)
- **Training Data:** ~2,180 Hindi conversational audio segments
- **Evaluation Set:** FLEURS Hindi (hi_in) — 418 test utterances

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-5 |
| Batch Size | 16 |
| Max Steps | 2,000 |
| Warmup Steps | 500 |
| Precision | FP16 |
| Gradient Checkpointing | Enabled |
| Best Model Selection | Lowest validation WER |
| GPU | Google Colab T4 (16GB VRAM) |

---

## 🔧 Post-Processing Pipeline

### Nasalization Restoration
Rule-based fix for the most common error type — dropped nasalization markers on oblique plurals (e.g., किसानो → किसानों).

| Metric | Result |
|--------|--------|
| Errors fixed | 15/21 (71.4%) |
| False positives | 0 |
| Utterances fully corrected | 6/10 |

### Number Normalization
Converts Hindi number words to digits while preserving idiomatic expressions.

| Input | Output | Type |
|-------|--------|------|
| दो सौ पचास रुपये | 250 रुपये | Compound |
| एक हज़ार रुपये | 1000 रुपये | Multiplier |
| दो-चार बातें | दो-चार बातें | Idiom preserved |

### English Word Detection
Tags English words in Hindi transcripts — handles both Latin script and Devanagari-transliterated English.

```
Input:  मेरा इंटरव्यू अच्छा गया और मुझे जॉब मिल गई
Output: मेरा [EN]इंटरव्यू[/EN] अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई
```

---

## 🚀 Deployment

Live on **HuggingFace Spaces** — [try it here](https://huggingface.co/spaces/rkyadav08/hindi-speech-to-text)

**Features:**
- **Quick Transcribe** — Upload audio or record from microphone
- **Dictation Mode** — Record multiple clips, accumulate text, copy into Word
- **Supported formats** — WAV, MP3, FLAC, OGG, M4A

---

## 💻 Run Locally

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/hindi-speech-recognition.git
cd hindi-speech-recognition

conda create -n whisper_hi python=3.11 -y
conda activate whisper_hi
pip install torch torchaudio transformers librosa soundfile gradio
```

### Transcribe an audio file

```bash
python transcribe.py "path/to/audio.wav"
```

### Run the web app locally

```bash
python app.py
# Opens at http://localhost:7860
```

### Run post-processing pipeline

```bash
python hindi_asr_pipeline.py --demo
```

---

## 📁 Project Structure

```
├── b_finetune_eval.py            # Fine-tuning & evaluation script
├── transcribe.py                 # Local inference script
├── app.py                        # Gradio web app
├── postprocess_hindi.py          # Nasalization restoration post-processor
├── hindi_asr_pipeline.py         # Number normalization + English detection
├── hindi_spelling_checker.py     # Spelling validation for 177K words
├── lattice_wer.py                # Lattice-based WER evaluation
├── processed_data/
│   ├── audio/                    # Training audio files
│   ├── train_manifest.json       # Training manifest
│   └── val_manifest.json         # Validation manifest
└── whisper-small-hi-finetuned/   # Fine-tuned model weights
    ├── model.safetensors
    ├── config.json
    ├── tokenizer.json
    ├── preprocessor_config.json
    └── ...
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | OpenAI Whisper-small (244M params) |
| Framework | PyTorch + HuggingFace Transformers |
| Training | Seq2SeqTrainer, FP16, Gradient Checkpointing |
| Compute | Google Colab T4 GPU (free tier) |
| Web App | Gradio |
| Deployment | HuggingFace Spaces |
| Post-Processing | Python regex, rule-based NLP |

---

## 📝 License

This project uses the OpenAI Whisper model under the MIT License.
