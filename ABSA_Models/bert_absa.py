import spacy
import pandas as pd
import torch
from transformers import pipeline
import ast  

nlp = spacy.load("en_core_web_trf")

bert_sentiment_pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

aspect_file = "/Users/busragural/Desktop/4.1/ABSA/Aspect Extraction/aspects.py"
with open(aspect_file, "r", encoding="utf-8") as f:
    aspects = ast.literal_eval(f.read().split("=", 1)[1].strip())

csv_file = "/Users/busragural/Desktop/4.1/Bitirme/gemini_ile_preprocess/preprocessingWithGemini.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]

def bert_sentiment_analysis(text):
    """BERT modelini kullanarak sentiment analizi yapar."""
    try:
        result = bert_sentiment_pipeline(text[:512])  # Max token limitini aşmamak için ilk 512 token alınıyor
        label = result[0]["label"]
        score = int(label.split()[0])  # Çıktı: '5 stars' formatında geliyor, ilk rakam alınıyor
        return score
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return 3  # Hata durumunda nötr (3 yıldız) dönüyoruz

def aspect_based_sentiment_analysis(text, aspects):
    """Belirlenen aspect'ler için BERT ile duygu analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores_bert = {aspect: [] for aspect in aspects}

    for sentence in doc.sents:  # Cümlelere ayır
        sentence_text = sentence.text.lower()  # Küçük harfe çevir
        for aspect, keywords in aspects.items():
            # Eğer listedeki herhangi bir kelime cümlede geçiyorsa aspect için duygu analizi yap
            if any(keyword.lower() in sentence_text for keyword in keywords):
                # BERT Sentiment Analizi
                bert_score = bert_sentiment_analysis(sentence.text)
                sentiment_scores_bert[aspect].append(bert_score)

    aspect_sentiments = {}
    valid_scores_bert = []

    for aspect in sentiment_scores_bert:
        avg_score_bert = sum(sentiment_scores_bert[aspect]) / len(sentiment_scores_bert[aspect]) if sentiment_scores_bert[aspect] else 0.0
        aspect_sentiments[f"{aspect}_bert_score"] = avg_score_bert
        aspect_sentiments[f"{aspect}_bert_stars"] = avg_score_bert  # BERT zaten 1-5 arasında döndürdüğü için doğrudan kullanılabilir
        
        if avg_score_bert != 0.0:
            valid_scores_bert.append(avg_score_bert)

    general_score_bert = sum(valid_scores_bert) / len(valid_scores_bert) if valid_scores_bert else 3  # Varsayılan nötr
    general_stars_bert = general_score_bert

    aspect_sentiments["general_bert_score"] = general_score_bert
    aspect_sentiments["general_bert_stars"] = general_stars_bert

    return aspect_sentiments

df = df[:5]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x, aspects))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "./Data/output/BERT_ABSA.csv"
df_with_sentiment.to_csv(output_file, index=False)

print("BERT ile ABSA işlemi tamamlandı ve sonuçlar kaydedildi!")
