import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import matplotlib.pyplot as plt
import io
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda" if torch.cuda.is_available() else "cpu")

def process_image_question(image_path, question):

    try:
        raw_image = Image.open(image_path).convert('RGB')
        
        inputs = processor(raw_image, question, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        out = model.generate(**inputs)
        
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        return answer
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def display_image(image_path):

    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

print("BLIP Görsel Soru-Cevap Sistemine Hoş Geldiniz!")
print("Çıkmak için 'q' girin.")
    
while True:
    image_path = input("\nGörsel dosya yolunu girin (veya sürükleyip bırakın): ").strip('"')
        
    if image_path.lower() == 'q':
        break
            
    if not os.path.exists(image_path):
        print("Hata: Dosya bulunamadı. Lütfen geçerli bir yol girin.")
        continue
            
    try:
        display_image(image_path)
        
        while True:
            question = input("\nGörsel hakkında bir soru sorun (çıkmak için 'q'): ")
            
            if question.lower() == 'q':
                break
                
            if not question.strip():
                print("Lütfen geçerli bir soru girin.")
                continue
                
            answer = process_image_question(image_path, question)
            print(f"\nCevap: {answer}")
            
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        continue

