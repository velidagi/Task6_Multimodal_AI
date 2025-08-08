import cv2
from mistralai import Mistral
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt


"""import cv2
import pytesseract

# Eğer Windows kullanıyorsan aşağıdaki satırın yorumunu kaldır:
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Görseli yükle
image = cv2.imread("image_to_text.jpg")

# Görüntüyü gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültü azaltma (isteğe bağlı)
gray = cv2.medianBlur(gray, 3)

# Basit eşikleme
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# OCR işlemi
text = pytesseract.image_to_string(thresh, lang="eng")  # Türkçe için: lang="tur"

print("Tespit edilen metin:")
print(text)
"""


# Mistral API anahtarınızı buraya ekleyin
api_key = "your-mistral-api-key"
client = Mistral(api_key=api_key)

def image_to_base64(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Base64'e çevir
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64

def extract_text_with_mistral(image):
    """Mistral OCR kullanarak metni çıkarır"""
    try:
        img_base64 = image_to_base64(image)
        
        response = client.chat.complete(
            model="pixtral-12b-2409",  
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Can you extract the text from this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_base64}"
                        }
                    ]
                }
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

image = cv2.imread("data/image_to_text.jpg")
if image is None:
    print("Görsel yüklenemedi! Dosya yolunu kontrol edin.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray, 3)

# Basit eşikleme
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# MistralOCR işlemi
text = extract_text_with_mistral(processed_image)

print("Tespit edilen metin:")
print(text)

