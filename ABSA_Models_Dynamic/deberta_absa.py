import spacy
import pandas as pd
import ast  
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_trf")

model_name = "yangheng/deberta-v3-large-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

embedding_model = SentenceTransformer('sentence-t5-xl')  

aspect_file = "/Users/busragural/Desktop/4.1/ABSA/Aspect Extraction/aspects.py"
with open(aspect_file, "r", encoding="utf-8") as f:
    aspect_definitions = ast.literal_eval(f.read().split("=", 1)[1].strip())


aspect_embeddings = {aspect: embedding_model.encode(description) for aspect, description in aspect_definitions.items()}

SIMILARITY_THRESHOLD = 0.72


def classify_aspects(sentence):
    """Cümledeki kelimeleri aspect'lerle eşleştirerek en uygun aspect'i bulur."""
    doc = nlp(sentence)
    word_aspects = {}

    for token in doc:
        if token.is_stop or token.is_punct:  
            continue

        word_vector = embedding_model.encode(token.text)  
        aspect_scores = {aspect: cosine_similarity([word_vector], [aspect_vector])[0][0] 
                         for aspect, aspect_vector in aspect_embeddings.items()}

        print(f"\nKelime: {token.text}")
        for aspect, score in aspect_scores.items():
            print(f"   {aspect}: {score:.4f}")

        filtered_scores = {aspect: score for aspect, score in aspect_scores.items() if score >= SIMILARITY_THRESHOLD}

        if not filtered_scores:
            continue

        min_score = min(filtered_scores.values())
        max_score = max(filtered_scores.values())

        if max_score - min_score > 0:
            normalized_scores = {aspect: (score - min_score) / (max_score - min_score) 
                                 for aspect, score in filtered_scores.items()}
        else:
            normalized_scores = filtered_scores  

        best_aspect = max(normalized_scores, key=normalized_scores.get)
        best_score = normalized_scores[best_aspect]

        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        second_best_aspect = sorted_scores[1][0] if len(sorted_scores) > 1 else None
        second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

        if second_best_aspect and best_score - second_best_score < 0.1:
            if best_score > SIMILARITY_THRESHOLD:
                word_aspects[best_aspect] = word_aspects.get(best_aspect, []) + [token.text]
            if second_best_score > SIMILARITY_THRESHOLD:
                word_aspects[second_best_aspect] = word_aspects.get(second_best_aspect, []) + [token.text]
        else:
            if best_score > SIMILARITY_THRESHOLD:
                word_aspects[best_aspect] = word_aspects.get(best_aspect, []) + [token.text]

    return word_aspects


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

def aspect_based_sentiment_analysis(text):
    """Aspect bazlı sentiment analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores = {aspect: [] for aspect in aspect_definitions}
    
    for sentence in doc.sents:
        detected_aspects = classify_aspects(sentence.text)  
        print(f"\nCümle: {sentence.text}")
        print(f"Aspect Eşleşmeleri: {detected_aspects}")

        for aspect, words in detected_aspects.items():
            sentiment, score = analyze_sentiment(sentence.text, aspect)
            print(f"Aspect: {aspect} - Sentiment Score: {score} - Kelimeler: {words}")

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


csv_file = "/Users/busragural/Desktop/4.1/Bitirme/gemini_ile_preprocess/preprocessingWithGemini.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]


df = df[:5]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "/Users/busragural/Desktop/4.1/ABSA/Data/output/DeBERTa_ABSA2.csv"
df_with_sentiment.to_csv(output_file, index=False)

print(f"Sonuçlar kaydedildi: {output_file}")

