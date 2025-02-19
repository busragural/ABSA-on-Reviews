import spacy
import pandas as pd
import ast  
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_trf")

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
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

def aspect_based_sentiment_analysis(text):
    """Belirlenen aspect'ler için duygu analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores = {aspect: [] for aspect in aspect_definitions}
    
    for sentence in doc.sents:
        detected_aspects = classify_aspects(sentence.text)
        print(f"\nCümle: {sentence.text}")
        print(f"Aspect Eşleşmeleri: {detected_aspects}")  
        for aspect, words in detected_aspects.items():
            score = get_sentiment_score(sentence.text)
            print(f"Aspect: {aspect} - Sentiment Score: {score} - Kelimeler: {words}")
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



csv_file = "/Users/busragural/Desktop/4.1/Bitirme/gemini_ile_preprocess/preprocessingWithGemini.csv"
df = pd.read_csv(csv_file)

df = df[df["Split_Review"].notna()]

df = df[:5]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "./Data/output/DistilBERT_ABSA2.csv"
df_with_sentiment.to_csv(output_file, index=False)

print(f"Analiz tamamlandı, sonuçlar {output_file} dosyasına kaydedildi.")