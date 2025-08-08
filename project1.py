import cv2
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
from typing import List, Tuple, Dict
import json

class MultimodalBackpackExtractor:
    """
    Multimodal AI ile çanta segmentasyonu
    
    Kullanılan Modeller:
    1. CLIP - Görüntü-metin eşleştirme
    2. SAM - Segment Anything Model
    3. BLIP - Görüntü açıklama
    """
    
    def __init__(self, sam_checkpoint_path="sam_vit_h_4b8939.pth"):
        """Multimodal modelleri yükle"""
        print("🤖 Multimodal AI modelleri yükleniyor...")
        
        # 1. CLIP Model (Görüntü-Metin Eşleştirme)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print("✅ CLIP modeli yüklendi")
        
        # 2. SAM Model (Segmentasyon)
        try:
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
            print("✅ SAM modeli yüklendi")
        except:
            print("⚠️ SAM modeli yüklenemedi. SAM checkpoint dosyasını indirin.")
            self.sam_predictor = None
            self.mask_generator = None
        
        # 3. BLIP Model (Görüntü Açıklama)
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            print("✅ BLIP modeli yüklendi")
        except:
            print("⚠️ BLIP modeli yüklenemedi")
            self.blip_processor = None
            self.blip_model = None
        
        # Çanta ile ilgili text prompt'ları
        self.backpack_prompts = [
            "a backpack",
            "a bag on someone's back",
            "a rucksack",
            "a school bag",
            "a travel backpack",
            "a person wearing a backpack"
        ]
    
    def analyze_image_with_blip(self, image_path: str) -> str:
        """BLIP ile görüntüyü analiz et ve açıklama üret"""
        if self.blip_processor is None:
            return "BLIP model not available"
        
        image = Image.open(image_path).convert('RGB')
        
        # BLIP ile görüntü açıklaması
        inputs = self.blip_processor(image, return_tensors="pt")
        out = self.blip_model.generate(**inputs, max_length=50)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    
    def find_backpack_with_clip(self, image_path: str) -> Dict:
        """CLIP kullanarak çanta tespiti"""
        image = Image.open(image_path).convert('RGB')
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Text prompt'ları tokenize et
        text_inputs = clip.tokenize(self.backpack_prompts).to(self.device)
        
        # Özellik çıkarma
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            
            # Benzerlik skorları
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarities[0].topk(3)
        
        # En yüksek skorlu prompt'ları döndür
        results = []
        for i, (value, idx) in enumerate(zip(values, indices)):
            results.append({
                "prompt": self.backpack_prompts[idx],
                "confidence": value.item(),
                "rank": i + 1
            })
        
        return {
            "analysis": results,
            "max_confidence": results[0]["confidence"],
            "best_match": results[0]["prompt"]
        }
    
    def segment_with_sam(self, image_path: str) -> List[Dict]:
        """SAM ile otomatik segmentasyon"""
        if self.mask_generator is None:
            return []
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # SAM ile tüm segmentleri üret
        masks = self.mask_generator.generate(image_rgb)
        
        # Maskeleri boyut ve kaliteye göre sırala
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def filter_backpack_segments(self, image_path: str, masks: List[Dict], clip_threshold: float = 0.15) -> List[Dict]:
        """CLIP skorlarına göre çanta segmentlerini filtrele"""
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        backpack_segments = []
        
        for i, mask_data in enumerate(masks[:20]):  # İlk 20 segmenti kontrol et
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']
            
            # Segment çok küçükse atla
            if mask_data['area'] < (height * width * 0.01):  # %1'den küçük
                continue
            
            # Maskelenmiş bölgeyi çıkar
            x, y, w, h = map(int, bbox)
            
            # Bounding box sınırlarını kontrol et
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w <= 0 or h <= 0:
                continue
            
            # Segment bölgesini kırp
            segment_region = image[y:y+h, x:x+w]
            
            if segment_region.size == 0:
                continue
            
            # Geçici dosya olarak kaydet
            temp_path = f"temp_segment_{i}.jpg"
            cv2.imwrite(temp_path, segment_region)
            
            try:
                # CLIP ile bu segmentin çanta olup olmadığını kontrol et
                clip_result = self.find_backpack_with_clip(temp_path)
                
                if clip_result["max_confidence"] > clip_threshold:
                    backpack_segments.append({
                        "mask": mask,
                        "bbox": bbox,
                        "area": mask_data['area'],
                        "clip_confidence": clip_result["max_confidence"],
                        "clip_match": clip_result["best_match"],
                        "segment_id": i
                    })
                
                # Geçici dosyayı sil
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Segment {i} işlenirken hata: {e}")
                continue
        
        # CLIP skoruna göre sırala
        backpack_segments.sort(key=lambda x: x['clip_confidence'], reverse=True)
        
        return backpack_segments
    
    def create_final_mask(self, image_path: str, backpack_segments: List[Dict]) -> np.ndarray:
        """En iyi çanta segmentlerinden final mask oluştur"""
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        final_mask = np.zeros((height, width), dtype=np.uint8)
        
        # En yüksek skorlu segmentleri birleştir
        for segment in backpack_segments[:2]:  # En iyi 2 segmenti al
            mask = segment['mask'].astype(np.uint8) * 255
            final_mask = cv2.bitwise_or(final_mask, mask)
        
        return final_mask
    
    def process_multimodal(self, image_path: str, output_path: str = "multimodal_canta.png") -> Dict:
        """Multimodal AI pipeline ile çanta çıkarma"""
        print(f"🎯 Multimodal AI ile analiz başlıyor: {image_path}")
        
        results = {
            "image_path": image_path,
            "output_path": output_path,
            "steps": {}
        }
        
        # 1. BLIP ile görüntü açıklaması
        print("📝 BLIP ile görüntü açıklanıyor...")
        blip_caption = self.analyze_image_with_blip(image_path)
        results["steps"]["blip_caption"] = blip_caption
        print(f"BLIP Açıklaması: {blip_caption}")
        
        # 2. CLIP ile çanta varlığı kontrolü
        print("🔍 CLIP ile çanta aranıyor...")
        clip_result = self.find_backpack_with_clip(image_path)
        results["steps"]["clip_analysis"] = clip_result
        print(f"CLIP En İyi Eşleşme: {clip_result['best_match']} ({clip_result['max_confidence']:.3f})")
        
        # 3. SAM ile segmentasyon
        print("✂️ SAM ile segmentasyon yapılıyor...")
        all_masks = self.segment_with_sam(image_path)
        results["steps"]["total_segments"] = len(all_masks)
        print(f"Toplam {len(all_masks)} segment bulundu")
        
        # 4. CLIP ile segment filtreleme
        print("🎯 Çanta segmentleri filtreleniyor...")
        backpack_segments = self.filter_backpack_segments(image_path, all_masks)
        results["steps"]["backpack_segments"] = len(backpack_segments)
        results["steps"]["best_segments"] = [
            {
                "clip_confidence": seg["clip_confidence"],
                "clip_match": seg["clip_match"],
                "area": seg["area"]
            }
            for seg in backpack_segments[:3]
        ]
        
        if not backpack_segments:
            print("❌ Çanta segmenti bulunamadı!")
            results["success"] = False
            return results
        
        print(f"✅ {len(backpack_segments)} çanta segmenti bulundu")
        
        # 5. Final mask oluşturma
        print("🎨 Final mask oluşturuluyor...")
        final_mask = self.create_final_mask(image_path, backpack_segments)
        
        # 6. Çantayı çıkarma
        image = cv2.imread(image_path)
        
        # Mask uygula
        result_image = cv2.bitwise_and(image, image, mask=final_mask)
        
        # Alpha channel ekle (şeffaf arka plan)
        result_rgba = cv2.cvtColor(result_image, cv2.COLOR_BGR2BGRA)
        result_rgba[:, :, 3] = final_mask  # Alpha channel = mask
        
        # Kaydet
        cv2.imwrite(output_path, result_rgba)
        
        results["success"] = True
        results["final_confidence"] = backpack_segments[0]["clip_confidence"]
        
        print(f"💾 Sonuç kaydedildi: {output_path}")
        
        return results
    
    def visualize_analysis(self, image_path: str, results: Dict, save_path: str = "analysis_visualization.png"):
        """Analiz sonuçlarını görselleştir"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Orijinal görüntü
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title("Orijinal Görüntü")
        axes[0, 0].axis('off')
        
        # CLIP analizi
        clip_data = results["steps"]["clip_analysis"]["analysis"]
        prompts = [item["prompt"] for item in clip_data]
        confidences = [item["confidence"] for item in clip_data]
        
        axes[0, 1].bar(range(len(prompts)), confidences)
        axes[0, 1].set_title("CLIP Güven Skorları")
        axes[0, 1].set_xticks(range(len(prompts)))
        axes[0, 1].set_xticklabels(prompts, rotation=45, ha='right')
        axes[0, 1].set_ylabel("Güven Skoru")
        
        # Sonuç görüntüsü
        if results["success"]:
            result_img = cv2.imread(results["output_path"], cv2.IMREAD_UNCHANGED)
            if result_img.shape[2] == 4:  # RGBA
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGRA2RGB)
            else:
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            axes[1, 0].imshow(result_rgb)
            axes[1, 0].set_title("Çıkarılan Çanta")
        else:
            axes[1, 0].text(0.5, 0.5, "Çanta Bulunamadı", ha='center', va='center')
            axes[1, 0].set_title("Sonuç")
        axes[1, 0].axis('off')
        
        # İstatistikler
        stats_text = f"""
        BLIP Açıklaması: {results['steps']['blip_caption']}
        
        Toplam Segment: {results['steps']['total_segments']}
        Çanta Segmenti: {results['steps']['backpack_segments']}
        
        En İyi Eşleşme: {results['steps']['clip_analysis']['best_match']}
        Güven Skoru: {results['steps']['clip_analysis']['max_confidence']:.3f}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        axes[1, 1].set_title("Analiz İstatistikleri")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Analiz görseli kaydedildi: {save_path}")

def download_sam_checkpoint():
    """SAM model checkpoint'ini indir"""
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    filename = "sam_vit_h_4b8939.pth"
    
    print("📥 SAM checkpoint indiriliyor... (Bu biraz zaman alabilir)")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rİndiriliyor: {percent:.1f}%", end="")
    
    print(f"\n✅ SAM checkpoint indirildi: {filename}")
    return filename

# Ana kullanım fonksiyonu
def extract_backpack_multimodal(image_path: str, output_path: str = "multimodal_canta.png"):
    """Multimodal AI ile çanta çıkarma - Basit kullanım"""
    
    # SAM checkpoint kontrolü
    checkpoint_path = "sam_vit_h_4b8939.pth"
    import os
    if not os.path.exists(checkpoint_path):
        print("SAM checkpoint bulunamadı, indiriliyor...")
        checkpoint_path = download_sam_checkpoint()
    
    # Multimodal extractor oluştur
    extractor = MultimodalBackpackExtractor(checkpoint_path)
    
    # İşlemi çalıştır
    results = extractor.process_multimodal(image_path, output_path)
    
    # Sonuçları görselleştir
    extractor.visualize_analysis(image_path, results)
    
    return results

# Akademik Değerlendirme Raporu
def generate_academic_report(results: Dict) -> str:
    """Akademik rapor oluştur"""
    report = f"""
# Multimodal AI ile Çanta Segmentasyonu - Akademik Rapor

## 1. Kullanılan Modeller

### 1.1 CLIP (Contrastive Language-Image Pre-training)
- **Amaç**: Görüntü-metin eşleştirme ile çanta tespiti
- **Güven Skoru**: {results['steps']['clip_analysis']['max_confidence']:.3f}
- **En İyi Eşleşme**: {results['steps']['clip_analysis']['best_match']}

### 1.2 BLIP (Bootstrapping Language-Image Pre-training)
- **Amaç**: Görüntü açıklaması ve sahne anlama
- **Açıklama**: {results['steps']['blip_caption']}

### 1.3 SAM (Segment Anything Model)
- **Amaç**: Universal segmentasyon
- **Toplam Segment**: {results['steps']['total_segments']}
- **Çanta Segmenti**: {results['steps']['backpack_segments']}

## 2. Multimodal Pipeline

1. **Vision-Language Understanding**: CLIP ve BLIP ile görüntü analizi
2. **Zero-shot Detection**: Önceden eğitilmemiş nesne tespiti
3. **Semantic Segmentation**: SAM ile piksel seviyesi ayırma
4. **Cross-modal Filtering**: Text prompt'lar ile segment filtreleme

## 3. Sonuçlar

- **Başarı Durumu**: {'Başarılı' if results['success'] else 'Başarısız'}
- **Final Güven Skoru**: {results.get('final_confidence', 0):.3f}
- **Çıktı Dosyası**: {results['output_path']}

## 4. Akademik Katkı

Bu proje multimodal AI'ın pratik uygulamasını göstermektedir:
- Vision-Language modellerinin entegrasyonu
- Zero-shot learning yaklaşımı
- Cross-modal attention mekanizması
"""
    
    return report

# Ana kullanım
if __name__ == "__main__":
    print("🎓 Multimodal AI ile Çanta Segmentasyonu - Akademik Proje")
    print("=" * 60)
    
    # Görüntü yolu
    image_path = "data/canta.jpg"  # Kendi fotoğrafınızın yolu
    
    try:
        # Multimodal analiz
        results = extract_backpack_multimodal(image_path)
        
        # Akademik rapor
        report = generate_academic_report(results)
        
        # Raporu kaydet
        with open("academic_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("\n📋 Akademik rapor 'academic_report.md' dosyasına kaydedildi")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        print("\n💡 Gerekli kütüphaneleri yüklediğinizden emin olun:")
        print("pip install torch torchvision clip-by-openai segment-anything transformers matplotlib")