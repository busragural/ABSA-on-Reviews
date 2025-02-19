import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from scipy.spatial.distance import euclidean

# NLP modeli ve sentiment analyzer yükle
nlp = spacy.load("en_core_web_trf")
analyzer = SentimentIntensityAnalyzer()
model = SentenceTransformer('sentence-t5-xl')  

# daha iyi embedding hesaplamak için)
aspect_definitions = {
    'food': "Food, taste, chicken, biryani, veg, paneer, dish, rice, menu, item, fry, spicy, sweet, butter, sauce, cheese, starter, main, course, buffet, drink, delicious, meal, cuisine, ingredient, flavor, portion, fresh, seasoning, presentation, quantity and tasty.",
    'ambiance': "Ambience, music, ambience, people, table, atmosphere, environment, light, seating, comfort, noise, temperature, ventilation, space, experience and decor.",
    'service': "Service, staff, waiter, customer, delivery, order, serve, wait, behavior, hospitality, polite, response, accuracy, assistance, queue, packaging, speed and friendly.",
    'cleanliness': "Clean, hygiene, safety, utensils, dust, dirt, bug, stain, spot, sanitation and sanitary.",
    'price': "Price, money, affordability, expensive, overpriced, cheap, discounts, offers, budget and value."
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

def aspect_based_sentiment_analysis(text):
    """Aspect bazlı duygu analizi yapar ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores = {aspect: [] for aspect in aspect_definitions}

    for sentence in doc.sents:
        detected_aspects = classify_aspects(sentence.text)  

        print(f"\nCümle: {sentence.text}")
        print(f"Aspect Eşleşmeleri: {detected_aspects}")

        for aspect, words in detected_aspects.items():
            scores = analyzer.polarity_scores(sentence.text) 
            print(f"Aspect: {aspect} - Sentiment Score: {scores['compound']} - Kelimeler: {words}")

            sentiment_scores[aspect].append(scores['compound'])

    print("\n⚡ Sentiment Skorları (Ön Temizlik):", sentiment_scores)

    aspect_sentiments = {}
    valid_scores = []  

    for aspect in sentiment_scores:
        valid_aspect_scores = [score for score in sentiment_scores[aspect] if score != 0.0]

        avg_score = sum(valid_aspect_scores) / len(valid_aspect_scores) if valid_aspect_scores else 0.0
        aspect_sentiments[f"{aspect}_score"] = avg_score
        aspect_sentiments[f"{aspect}_stars"] = sentiment_to_star(avg_score) 

        if avg_score != 0.0:
            valid_scores.append(avg_score)

    general_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    general_stars = sentiment_to_star(general_score)
    aspect_sentiments["general_score"] = general_score
    aspect_sentiments["general_stars"] = general_stars

    return aspect_sentiments

'''
sentence = "Came here for lunch. The food was good and tasty. We tried buffet. All the items in veg and non-veg were tasty. The ambience was good. The service was not good because we wait too much. The staff were courteous. Meals price was very expensive."
result = aspect_based_sentiment_analysis(sentence)
print("\nSonuç:", result)
'''


csv_file = "/Users/busragural/Desktop/4.1/ABSA/Data/output/data_for_vader.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]

df = df[:15]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "./Data/output/Vader_ABSA2.csv"
df_with_sentiment.to_csv(output_file, index=False)