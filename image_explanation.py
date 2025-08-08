import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image_caption(image_path):

    try:
        raw_image = Image.open(image_path).convert('RGB')
        
        inputs = processor(raw_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        out = model.generate(**inputs)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def display_image_with_caption(image_path, caption):

    img = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Açıklama: {caption}", wrap=True)
    plt.axis('off')
    plt.show()

def main():
    print("BLIP Görüntü Açıklama Sistemine Hoş Geldiniz!")
    
    while True:
        # Görsel 
        image_path = input("\nGörsel dosya yolunu girin (veya sürükleyip bırakın) 'q': ").strip('"')
        
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            print("Hata: Dosya bulunamadı. Lütfen geçerli bir yol girin.")
            continue
            
        try:
            # Görsel için açıklama
            #print(f"\n'{image_path}' görseli işleniyor...")
            caption = generate_image_caption(image_path)
            #print(f"\nOluşturulan Açıklama: {caption}")
            
            display_image_with_caption(image_path, caption)
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            continue

if __name__ == "__main__":
    main()