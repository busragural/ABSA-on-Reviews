import spacy
import pandas as pd
import torch
from transformers import pipeline
import ast  
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_trf")

roberta_sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

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


def aspect_based_sentiment_analysis(text):
    """Belirlenen aspect'ler için RoBERTa ile duygu analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores_roberta ={aspect: [] for aspect in aspect_definitions}
    sentiment_confidences_roberta = {aspect: [] for aspect in aspect_definitions}

    for sentence in doc.sents:  # Cümlelere ayır
        detected_aspects = classify_aspects(sentence.text)
        print(f"\nCümle: {sentence.text}")
        print(f"Aspect Eşleşmeleri: {detected_aspects}")  
        
        for aspect, words in detected_aspects.items():
            roberta_stars, roberta_score = roberta_sentiment_analysis(sentence.text)
            print(f"Aspect: {aspect} - Sentiment Score: {roberta_score} - Kelimeler: {words}")
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



csv_file = "/Users/busragural/Desktop/4.1/Bitirme/gemini_ile_preprocess/preprocessingWithGemini.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]

df = df[:5]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "./Data/output/RoBERTa_ABSA2.csv"
df_with_sentiment.to_csv(output_file, index=False)