import cv2
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def video_karelerini_ayikla(video_path, output_folder, frame_rate=1):
    """
    Belirtilen video dosyasından saniyede belirli bir sayıda kare ayıklar ve kaydeder.
    (Bu fonksiyon önceki cevap ile aynıdır.)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' klasörü oluşturuldu.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: '{video_path}' video dosyası açılamadı.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print("Hata: Video FPS değeri okunamadı. Varsayılan olarak 30 kabul ediliyor.")
        video_fps = 30
        
    skip_interval = int(video_fps / frame_rate)
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_interval == 0:
            output_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(output_filename, frame)
            # print(f"'{output_filename}' kaydedildi.") # Analiz sırasında çıktıyı temiz tutmak için kapatıldı
            saved_frame_count += 1
        
        frame_count += 1

    cap.release()
    print(f"\nKare ayıklama tamamlandı. Toplam {saved_frame_count} adet kare '{output_folder}' klasörüne kaydedildi.")
    return True

def kareleri_llm_ile_analiz_et(input_folder, result_file):
    """
    Belirtilen klasördeki kareleri bir Vision-Language Model (BLIP) kullanarak analiz eder
    ve her kare için bir açıklama metni üretir.
    """
    print("\n--- LLM ile Analiz Başlatılıyor ---")
    
    # GPU varsa GPU'yu, yoksa CPU'yu kullan
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan cihaz: {device}")

    # Modeli ve işlemciyi Hugging Face'den yükle
    # Bu işlem model dosyalarını indireceği için ilk çalıştırmada zaman alabilir.
    print("BLIP modeli ve işlemcisi yükleniyor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    print("Model başarıyla yüklendi.")

    # Analiz edilecek görüntü dosyalarını bul
    try:
        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"Hata: '{input_folder}' klasörü bulunamadı.")
        return

    # Sonuçları kaydetmek için bir metin dosyası aç
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("Video Karesi Analiz Sonuçları (BLIP Modeli)\n")
        f.write("="*40 + "\n")
        
        # Her bir kare için analiz yap
        for filename in image_files:
            image_path = os.path.join(input_folder, filename)
            
            # Görüntüyü PIL formatında aç (transformers kütüphanesi bunu bekler)
            raw_image = Image.open(image_path).convert('RGB')
            
            # Görüntüyü modele uygun formata getir
            inputs = processor(images=raw_image, return_tensors="pt").to(device)
            
            # Modelden açıklama metnini üretmesini iste
            out = model.generate(**inputs, max_new_tokens=50) # max_new_tokens ile üretilecek metnin uzunluğunu sınırlayabiliriz
            
            # Üretilen token'ları okunabilir metne çevir
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # Sonuçları ekrana ve dosyaya yazdır
            print(f"[{filename}] -> {caption}")
            f.write(f"Dosya: {filename}\nAçıklama: {caption}\n---\n")

    print(f"\nAnaliz tamamlandı. Sonuçlar '{result_file}' dosyasına kaydedildi.")


# --- ANA KOD ---
if __name__ == "__main__":
    # Parametreleri Ayarla
    # NOT: 'test_video.mp4' adında bir videonun kodla aynı dizinde olduğunu varsayın.
    # Bu ismi kendi video dosyanızın adıyla değiştirin.
    video_dosyasi = "data/dia0_utt12.mp4" 
    ayiklanan_kareler_klasoru = "kaydedilen_kareler_llm"
    analiz_sonuc_dosyasi = "frame_log.txt"
    
    # Adım 1: Videodan kareleri ayıkla (saniyede 1 kare)
    if video_karelerini_ayikla(
        video_path=video_dosyasi, 
        output_folder=ayiklanan_kareler_klasoru, 
        frame_rate=2  # Videonun her saniyesinden x kare al
    ):
        # Adım 2: Ayıklanan kareleri LLM ile analiz et
        kareleri_llm_ile_analiz_et(
            input_folder=ayiklanan_kareler_klasoru,
            result_file=analiz_sonuc_dosyasi
        )