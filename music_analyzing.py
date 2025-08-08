import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

model_name = "dima806/music_genres_classification"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)


genres = ["blues", "classical", "country","disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

mp3_file = "data/Michael Jackson - Billie Jean (Official Video) [Zi_XLOBDo_Y].mp3"

audio, sr = librosa.load(mp3_file, sr=16000, duration=30)

print(audio.shape)
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
print(inputs)

with torch.no_grad():
   outputs = model(inputs['input_values'])
   predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
   print(predictions)
# result:
predicted_id = torch.argmax(predictions, dim=-1).item()
print("predictedid:",predicted_id)

predicted_genre = genres[predicted_id]
confidence = predictions[0][predicted_id].item() * 100

print(f"Dosya: {mp3_file}")
print(f"Tahmin edilen tür: {predicted_genre}")
print(f"Güven: {confidence:.2f}%")