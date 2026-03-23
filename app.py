import torch
import gradio as gr
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_PATH = "./"

print("Loading model...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")


def transcribe_audio(audio_input):
    if audio_input is None:
        return "No audio provided."
    try:
        if isinstance(audio_input, tuple):
            sr, audio_data = audio_input
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        else:
            audio_data, sr = librosa.load(audio_input, sr=16000)
        chunk_length = 30 * 16000
        chunks = [audio_data[i:i + chunk_length] for i in range(0, len(audio_data), chunk_length)]
        full_transcription = []
        for chunk in chunks:
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(inputs, max_new_tokens=440, language="hi", task="transcribe")
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            full_transcription.append(text.strip())
        result = " ".join(full_transcription)
        if not result.strip():
            return "No speech detected."
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def transcribe_and_append(audio_input, existing_text):
    new_text = transcribe_audio(audio_input)
    if new_text.startswith("No audio") or new_text.startswith("Error"):
        return existing_text, new_text
    if existing_text:
        return existing_text + " " + new_text, new_text
    return new_text, new_text


with gr.Blocks() as demo:
    gr.Markdown("# Hindi Speech-to-Text")
    gr.Markdown("Fine-tuned Whisper model for accurate Hindi transcription")
    with gr.Tabs():
        with gr.TabItem("Quick Transcribe"):
            gr.Markdown("Upload an audio file or record from your microphone.")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_upload = gr.Audio(label="Upload Audio or Record", type="filepath", sources=["upload", "microphone"])
                    transcribe_btn = gr.Button("Transcribe", variant="primary")
                with gr.Column(scale=1):
                    output_text = gr.Textbox(label="Transcription", placeholder="Hindi transcription will appear here...", lines=8)
            transcribe_btn.click(fn=transcribe_audio, inputs=[audio_upload], outputs=[output_text])
        with gr.TabItem("Dictation Mode"):
            gr.Markdown("Record multiple clips. Each one appends to your document.")
            dictation_audio = gr.Audio(label="Record a clip", type="numpy", sources=["microphone"])
            with gr.Row():
                dictate_btn = gr.Button("Add to Document", variant="primary")
                clear_btn = gr.Button("Clear All", variant="secondary")
            last_segment = gr.Textbox(label="Last segment", interactive=False, lines=2)
            full_document = gr.Textbox(label="Full Document (copy into Word)", placeholder="Start dictating...", lines=12)
            dictate_btn.click(fn=transcribe_and_append, inputs=[dictation_audio, full_document], outputs=[full_document, last_segment])
            clear_btn.click(fn=lambda: ("", ""), outputs=[full_document, last_segment])
    gr.Markdown("**Supported formats:** WAV, MP3, FLAC, OGG, M4A")

if __name__ == "__main__":
    demo.launch()