import spacy
import pandas as pd
import ast  
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nlp = spacy.load("en_core_web_trf")

model_name = "yangheng/deberta-v3-large-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

aspect_file = "/Users/busragural/Desktop/4.1/ABSA/Aspect Extraction/aspects.py"
with open(aspect_file, "r", encoding="utf-8") as f:
    aspects = ast.literal_eval(f.read().split("=", 1)[1].strip())

csv_file = "/Users/busragural/Desktop/4.1/ABSA/Data/output/preproces_for_vader2.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]

def sentiment_to_star(label, score):
    """Sentiment label'ına ve olasılık skoruna göre 1 ile 5 arasında yıldız verir."""
    if label == "Positive":
        if score >= 0.8:
            return 5
        elif score >= 0.6:
            return 4
        elif score >= 0.4:
            return 3
        elif score >= 0.2:
            return 2
        else:
            return 1  # Çok düşük pozitiflik
    elif label == "Negative":
        if score >= 0.8:
            return 1  # Çok güçlü negatif
        elif score >= 0.6:
            return 2
        elif score >= 0.4:
            return 3
        elif score >= 0.2:
            return 4
        else:
            return 5  # Negatifliği düşükse daha olumlu olur
    else:  # Neutral
        return 3  # Nötr olan her şey için orta değer

def analyze_sentiment(text, aspect):
    """DeBERTa modelini kullanarak sentiment analizi yapar."""
    inputs = tokenizer(f"{aspect} [SEP] {text}", return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
    
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    predicted_class = torch.argmax(probabilities).item()
    sentiment = sentiment_labels[predicted_class]
    sentiment_score = probabilities[predicted_class].item()

    return sentiment, sentiment_score

def aspect_based_sentiment_analysis(text, aspects):
    """Aspect bazlı sentiment analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores = {aspect: [] for aspect in aspects}
    
    for sentence in doc.sents:  # Cümlelere ayır
        sentence_text_lower = sentence.text.lower()  # Küçük harfe çevir
        for aspect, keywords in aspects.items():
            if any(keyword.lower() in sentence_text_lower for keyword in keywords):
                sentiment, score = analyze_sentiment(sentence.text, aspect)  # Sentiment analizi yap
                sentiment_scores[aspect].append((sentiment, score))

    aspect_sentiments = {}
    valid_scores = []
    
    for aspect in sentiment_scores:
        scores = [s[1] for s in sentiment_scores[aspect] if s]  # Sadece skorları al
        labels = [s[0] for s in sentiment_scores[aspect] if s]  # Sadece etiketleri al

        if scores:
            avg_score = sum(scores) / len(scores)
            dominant_label = max(set(labels), key=labels.count)  # En sık geçen sentiment label
            stars = sentiment_to_star(dominant_label, avg_score)  # Yıldız hesapla
        else:
            avg_score = 0.0
            dominant_label = None  # Aspect geçmiyorsa herhangi bir etiket atanmaz
            stars = 0  # Eğer aspect geçmiyorsa yıldız da 0 olmalı

        aspect_sentiments[f"{aspect}_score"] = avg_score
        aspect_sentiments[f"{aspect}_stars"] = stars
        
        if avg_score != 0.0:
            valid_scores.append(avg_score)

    general_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    general_label = None
    if general_score > 0.6:
        general_label = "Positive"
    elif general_score < 0.4:
        general_label = "Negative"
    else:
        general_label = "Neutral"

    general_stars = sentiment_to_star(general_label, general_score) if general_label else 0

    aspect_sentiments["general_score"] = general_score
    aspect_sentiments["general_stars"] = general_stars

    return aspect_sentiments

df = df[:5]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x, aspects))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "/Users/busragural/Desktop/4.1/ABSA/Data/output/DeBERTa_ABSA.csv"
df_with_sentiment.to_csv(output_file, index=False)

print(f"Sonuçlar kaydedildi: {output_file}")
