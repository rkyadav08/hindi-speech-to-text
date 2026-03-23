"""
Hindi Speech-to-Text — Gradio App for HuggingFace Spaces
Deploy for FREE on https://huggingface.co/spaces
 
Features:
  - Upload audio files or record from microphone
  - Real-time transcription using your fine-tuned Whisper model
  - Copy button to paste into Word/Docs
  - Supports WAV, MP3, FLAC, OGG, M4A
"""
 
import torch
import gradio as gr
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
 
# ─── Load Model ───────────────────────────────────────────
# For HuggingFace Spaces: upload your model files to the Space repo
# For local use: point to your local model path
MODEL_PATH = "./"  # Current directory (model files in repo root)
# MODEL_PATH = "./whisper-small-hi-finetuned"  # Uncomment for local use
 
print("Loading model...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
 
 
# ─── Transcription Functions ──────────────────────────────
def transcribe_audio(audio_input):
    """Transcribe audio from file upload or microphone."""
    if audio_input is None:
        return "⚠️ No audio provided. Please upload a file or record from microphone."
 
    try:
        # Gradio returns (sample_rate, numpy_array) for microphone
        # or a file path for file upload
        if isinstance(audio_input, tuple):
            sr, audio_data = audio_input
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            # Convert stereo to mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            # Resample to 16kHz if needed
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        else:
            # File path
            audio_data, sr = librosa.load(audio_input, sr=16000)
 
        # Process in chunks for long audio (30 seconds each)
        chunk_length = 30 * 16000  # 30 seconds
        chunks = [audio_data[i:i + chunk_length] for i in range(0, len(audio_data), chunk_length)]
 
        full_transcription = []
        for chunk in chunks:
            inputs = processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)
 
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs,
                    max_new_tokens=448,
                    language="hi",
                    task="transcribe"
                )
 
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            full_transcription.append(text.strip())
 
        result = " ".join(full_transcription)
 
        if not result.strip():
            return "⚠️ No speech detected in the audio."
 
        return result
 
    except Exception as e:
        return f"❌ Error: {str(e)}"
 
 
def transcribe_and_append(audio_input, existing_text):
    """Transcribe and append to existing text (for continuous dictation)."""
    new_text = transcribe_audio(audio_input)
    if new_text.startswith("⚠️") or new_text.startswith("❌"):
        return existing_text, new_text
    if existing_text:
        combined = existing_text + " " + new_text
    else:
        combined = new_text
    return combined, new_text
 
 
# ─── Gradio Interface ────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600;700&family=DM+Sans:wght@400;500;700&display=swap');
 
.gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    max-width: 900px !important;
    margin: auto !important;
}
 
.main-title {
    text-align: center;
    font-size: 2.2em;
    font-weight: 700;
    margin-bottom: 0.2em;
    background: linear-gradient(135deg, #FF6B35, #F7C948);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
 
.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-bottom: 1.5em;
}
 
.transcription-box textarea {
    font-family: 'Noto Sans Devanagari', 'DM Sans', sans-serif !important;
    font-size: 1.15em !important;
    line-height: 1.8 !important;
    min-height: 200px !important;
}
 
footer { display: none !important; }
"""
 
with gr.Blocks(css=CUSTOM_CSS, title="Hindi Speech-to-Text") as demo:
 
    gr.HTML("""
        <div class="main-title">🎙️ Hindi Speech-to-Text</div>
        <div class="subtitle">Fine-tuned Whisper model for accurate Hindi transcription</div>
    """)
 
    with gr.Tabs():
 
        # ── Tab 1: Quick Transcribe ──
        with gr.TabItem("🎤 Quick Transcribe"):
            gr.Markdown("Upload an audio file or record from your microphone.")
 
            with gr.Row():
                with gr.Column(scale=1):
                    audio_upload = gr.Audio(
                        label="Upload Audio or Record",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    transcribe_btn = gr.Button(
                        "✨ Transcribe",
                        variant="primary",
                        size="lg"
                    )
 
                with gr.Column(scale=1):
                    output_text = gr.Textbox(
                        label="Transcription",
                        placeholder="Hindi transcription will appear here...",
                        lines=8,
                        show_copy_button=True,
                        elem_classes=["transcription-box"]
                    )
 
            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=[audio_upload],
                outputs=[output_text]
            )
 
        # ── Tab 2: Dictation Mode ──
        with gr.TabItem("📝 Dictation Mode"):
            gr.Markdown(
                "Record multiple clips — each one appends to your document. "
                "Copy the full text into Word when done."
            )
 
            dictation_audio = gr.Audio(
                label="Record a clip",
                type="numpy",
                sources=["microphone"]
            )
 
            with gr.Row():
                dictate_btn = gr.Button("➕ Add to Document", variant="primary")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
 
            last_segment = gr.Textbox(
                label="Last segment",
                interactive=False,
                lines=2
            )
 
            full_document = gr.Textbox(
                label="📄 Full Document (copy this into Word)",
                placeholder="Start dictating — your text will accumulate here...",
                lines=12,
                show_copy_button=True,
                elem_classes=["transcription-box"]
            )
 
            dictate_btn.click(
                fn=transcribe_and_append,
                inputs=[dictation_audio, full_document],
                outputs=[full_document, last_segment]
            )
 
            clear_btn.click(
                fn=lambda: ("", ""),
                outputs=[full_document, last_segment]
            )
 
    # ── Footer info ──
    gr.Markdown("""
    ---
    **How to use with Microsoft Word:**
    1. Transcribe your audio above
    2. Click the 📋 copy button on the text box
    3. Paste into Word (`Ctrl+V`)
 
    **Supported formats:** WAV, MP3, FLAC, OGG, M4A | **Model:** Fine-tuned Whisper-small for Hindi
    """)
 
# Launch
if __name__ == "__main__":
    demo.launch()
 