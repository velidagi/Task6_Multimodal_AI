import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from openai import OpenAI
import os
from dotenv import load_dotenv

# BLIP Modeli
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
api_key = os.getenv("OPENAI_V2")
client = OpenAI(api_key=api_key)  

def transcribe_audio(audio_path):
    """Ses dosyasını metne çevirir"""
    try:
        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return result
    except Exception as e:
        return f"Hata: {str(e)}"

def answer_question(image, question):
    """Görsel ve sorudan cevap üretir"""
    try:
        if image is None:
            return "Lütfen bir görsel yükleyin!"
        
        raw_image = Image.fromarray(image).convert('RGB')
        inputs = blip_processor(raw_image, question, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"Hata: {str(e)}"

# Gradio Arayüzü
with gr.Blocks(title="Multimodal AI Asistan", theme="soft") as app:
    gr.Markdown("""
    # 🚀 Whisper + BLIP Multimodal Asistan
    **Ses transkripsiyonu** ve **görsel soru-cevap** sistemi
    """)
    
    with gr.Tab("🎤 Ses Transkripsiyonu"):
        gr.Markdown("### Ses dosyanızı yükleyin (mp3, wav, vb.)")
        with gr.Row():
            audio_input = gr.Audio(label="Ses Kaydı", type="filepath")
            text_output = gr.Textbox(label="Transkripsiyon Sonucu", lines=5)
        transcribe_btn = gr.Button("Döküme Dönüştür", variant="primary")
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=audio_input,
            outputs=text_output,
            api_name="transcribe"
        )
    with gr.Tab("🖼️ Görsel Analiz (BLIP)"):
        with gr.Row():
            image_input = gr.Image(label="Görsel yükle")
            with gr.Column():
                question_input = gr.Textbox(label="Sorunuz", placeholder="Bu görselde ne görüyorum?")
                answer_output = gr.Textbox(label="Cevap", interactive=False)
        ask_btn = gr.Button("Soru Sor")
        ask_btn.click(answer_question, inputs=[image_input, question_input], outputs=answer_output)
        
        # Örnek Sorular
        gr.Examples(
            examples=[
                ["What color backpack does the girl have?", "data/canta.jpg"]
                ],
            inputs=[question_input, image_input],
            label="Örnek:"
        )

    

# Uygulamayı başlat
if __name__ == "__main__":
    app.launch(server_port=7860, share=True)