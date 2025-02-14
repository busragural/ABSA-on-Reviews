import spacy
import pandas as pd
import torch
from transformers import pipeline
import ast  

nlp = spacy.load("en_core_web_trf")

roberta_sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

aspect_file = "/Users/busragural/Desktop/4.1/ABSA/Aspect Extraction/aspects.py"
with open(aspect_file, "r", encoding="utf-8") as f:
    aspects = ast.literal_eval(f.read().split("=", 1)[1].strip())

csv_file = "/Users/busragural/Desktop/4.1/Bitirme/gemini_ile_preprocess/preprocessingWithGemini.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]

def roberta_sentiment_analysis(text):
    """RoBERTa modelini kullanarak sentiment analizi yapar ve 1-5 arasında yıldız değeri döndürür."""
    try:
        result = roberta_sentiment_pipeline(text[:512])  # 512 token limitini aşmamak için kesiyoruz
        label = result[0]["label"]  # Model çıktısı: 'POSITIVE' veya 'NEGATIVE'
        score = result[0]["score"]  # Modelin güven skoru (0 ile 1 arasında)

        if label == "POSITIVE":
            if score >= 0.8:
                return 5, score  # Çok güçlü pozitif
            elif score >= 0.6:
                return 4, score  # Orta derecede pozitif
            else:
                return 3, score  # Hafif pozitif
        elif label == "NEGATIVE":
            if score >= 0.8:
                return 1, -score  # Çok güçlü negatif
            elif score >= 0.6:
                return 2, -score  # Orta derecede negatif
            else:
                return 3, -score  # Hafif negatif

    except Exception as e:
        print(f"Hata oluştu: {e}")
        return 3, 0.0  


def aspect_based_sentiment_analysis(text, aspects):
    """Belirlenen aspect'ler için RoBERTa ile duygu analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores_roberta = {aspect: [] for aspect in aspects}
    sentiment_confidences_roberta = {aspect: [] for aspect in aspects}

    for sentence in doc.sents:  # Cümlelere ayır
        sentence_text = sentence.text.lower()  # Küçük harfe çevir
        for aspect, keywords in aspects.items():
            if any(keyword.lower() in sentence_text for keyword in keywords):
                # RoBERTa Sentiment Analizi
                roberta_stars, roberta_score = roberta_sentiment_analysis(sentence.text)
                sentiment_scores_roberta[aspect].append(roberta_stars)
                sentiment_confidences_roberta[aspect].append(roberta_score)

    aspect_sentiments = {}
    valid_scores_roberta = []

    for aspect in sentiment_scores_roberta:
        avg_stars_roberta = sum(sentiment_scores_roberta[aspect]) / len(sentiment_scores_roberta[aspect]) if sentiment_scores_roberta[aspect] else 0.0
        avg_score_roberta = sum(sentiment_confidences_roberta[aspect]) / len(sentiment_confidences_roberta[aspect]) if sentiment_confidences_roberta[aspect] else 0.0

        aspect_sentiments[f"{aspect}_roberta_score"] = avg_score_roberta  # Modelin döndürdüğü confidence score
        aspect_sentiments[f"{aspect}_roberta_stars"] = avg_stars_roberta  # 1-5 arası yıldız

        if avg_stars_roberta != 0.0:
            valid_scores_roberta.append(avg_stars_roberta)

    general_score_roberta = sum(valid_scores_roberta) / len(valid_scores_roberta) if valid_scores_roberta else 3  # Varsayılan nötr
    general_stars_roberta = general_score_roberta

    aspect_sentiments["general_roberta_score"] = general_score_roberta
    aspect_sentiments["general_roberta_stars"] = general_stars_roberta

    return aspect_sentiments


df = df[:10]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x, aspects))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "./Data/output/RoBERTa_ABSA.csv"
df_with_sentiment.to_csv(output_file, index=False)

print("RoBERTa ile ABSA işlemi tamamlandı ve sonuçlar kaydedildi!")
