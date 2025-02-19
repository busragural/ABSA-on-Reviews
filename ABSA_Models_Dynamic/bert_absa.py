import spacy
import pandas as pd
import torch
from transformers import pipeline
import ast  
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_trf")

bert_sentiment_pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
model = SentenceTransformer('sentence-t5-xl')  

aspect_definitions = {
    'food': "Food, taste, chicken, biryani, veg, paneer, dish, rice, menu, item, fry, spicy, sweet, butter, sauce, cheese, starter, quality, main, course, buffet, drink, delicious, meal, cuisine, ingredient, flavor, portion, fresh, seasoning, presentation, quantity and tasty.",
    'ambiance': "Ambience, music, ambience, people, look, table, atmosphere, environment, light, seating, comfort, noise, temperature, ventilation, space, experience and decor.",
    'service': "Service, staff, waiter, customer, delivery, order, serve, wait, give, take, recommend, behavior, hospitality, polite, response, accuracy, assistance, queue, packaging, speed and friendly.",
    'cleanliness': "Clean, hygiene, safety, utensils, dust, dirt, bug, stain, spot, sanitation and sanitary.",
    'price': "Price, money, worth, affordability, expensive, overpriced, cheap, discounts, offers, budget and value."
}

aspect_embeddings = {aspect: model.encode(description) for aspect, description in aspect_definitions.items()}

SIMILARITY_THRESHOLD = 0.72

def classify_aspects(sentence):
    """Cümledeki kelimeleri aspect'lerle eşleştirerek en uygun aspect'i bulur."""
    doc = nlp(sentence)
    word_aspects = {}

    for token in doc:
        if token.is_stop or token.is_punct:  
            continue

        word_vector = model.encode(token.text)  
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

def aspect_based_sentiment_analysis(text):
    """Belirlenen aspect'ler için BERT ile duygu analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores_bert = {aspect: [] for aspect in aspect_definitions}

    for sentence in doc.sents:  # Cümlelere ayır
        detected_aspects = classify_aspects(sentence.text)  
        print(f"\nCümle: {sentence.text}")
        print(f"Aspect Eşleşmeleri: {detected_aspects}")

        for aspect, words in detected_aspects.items():
            bert_score = bert_sentiment_analysis(sentence.text) 
            print(f"Aspect: {aspect} - Sentiment Score: {bert_score} - Kelimeler: {words}")

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


csv_file = "/Users/busragural/Desktop/4.1/Bitirme/gemini_ile_preprocess/preprocessingWithGemini.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]

df = df[:15]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "./Data/output/BERT_ABSA2.csv"
df_with_sentiment.to_csv(output_file, index=False)

print("BERT ile ABSA işlemi tamamlandı ve sonuçlar kaydedildi!")
