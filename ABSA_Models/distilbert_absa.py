import spacy
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ast  # Aspect dictionary'yi güvenli şekilde import etmek için

nlp = spacy.load("en_core_web_trf")  # Daha hızlı çalışması için küçük model

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


aspect_file = "/Users/busragural/Desktop/4.1/ABSA/Aspect Extraction/aspects.py"
with open(aspect_file, "r", encoding="utf-8") as f:
    aspects = ast.literal_eval(f.read().split("=", 1)[1].strip())

csv_file = "/Users/busragural/Desktop/4.1/Bitirme/gemini_ile_preprocess/preprocessingWithGemini.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]

def sentiment_to_star(score):
    """Sentiment skorunu 1 ile 5 arasında yıldız değerine dönüştür."""
    if score == 0.0:
        return 0  # Aspect geçmiyorsa 0 yıldız
    elif score <= -0.6:
        return 1
    elif score <= -0.2:
        return 2
    elif score <= 0.2:
        return 3
    elif score <= 0.6:
        return 4
    else:
        return 5

def get_sentiment_score(text):
    """DistilBERT ile cümleye duygu analizi yapar ve skor döndürür."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=-1)
        positive_score = scores[0][1].item()  # Pozitif duygu skoru
        negative_score = scores[0][0].item()  # Negatif duygu skoru
        compound_score = positive_score - negative_score  # VADER compound skoruna benzer bir skor üretelim
    return compound_score

def aspect_based_sentiment_analysis(text, aspects):
    """Belirlenen aspect'ler için duygu analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores = {aspect: [] for aspect in aspects}
    
    for sentence in doc.sents:  # Cümlelere ayır
        sentence_text_lower = sentence.text.lower()  # Küçük harfe çevir
        for aspect, keywords in aspects.items():
            # Eğer listedeki herhangi bir kelime cümlede geçiyorsa aspect için duygu analizi yap
            if any(keyword.lower() in sentence_text_lower for keyword in keywords):
                score = get_sentiment_score(sentence.text)  # DistilBERT ile sentiment analizi yap
                sentiment_scores[aspect].append(score)

    aspect_sentiments = {}
    valid_scores = []  
    
    for aspect in sentiment_scores:
        avg_score = sum(sentiment_scores[aspect]) / len(sentiment_scores[aspect]) if sentiment_scores[aspect] else 0.0
        aspect_sentiments[f"{aspect}_score"] = avg_score
        aspect_sentiments[f"{aspect}_stars"] = sentiment_to_star(avg_score)  # Yıldız hesapla
        
        if avg_score != 0.0:
            valid_scores.append(avg_score)

    general_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    general_stars = sentiment_to_star(general_score)

    aspect_sentiments["general_score"] = general_score
    aspect_sentiments["general_stars"] = general_stars

    return aspect_sentiments

df = df[:10]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x, aspects))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "./Data/output/DistilBERT_ABSA.csv"
df_with_sentiment.to_csv(output_file, index=False)

print(f"Analiz tamamlandı, sonuçlar {output_file} dosyasına kaydedildi.")
