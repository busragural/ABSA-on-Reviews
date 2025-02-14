import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ast  

nlp = spacy.load("en_core_web_trf")

analyzer = SentimentIntensityAnalyzer()

aspect_file = "/Users/busragural/Desktop/4.1/ABSA/Aspect Extraction/aspects.py"
with open(aspect_file, "r", encoding="utf-8") as f:
    aspects = ast.literal_eval(f.read().split("=", 1)[1].strip())

csv_file = "/Users/busragural/Desktop/4.1/ABSA/Data/output/preproces_for_vader2.csv"
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

def aspect_based_sentiment_analysis(text, aspects):
    """Belirlenen aspect'ler için duygu analizi yapar, yıldız derecelendirmesi ekler ve genel puanı hesaplar."""
    doc = nlp(text)
    sentiment_scores = {aspect: [] for aspect in aspects}
    
    for sentence in doc.sents:  # Cümlelere ayır
        sentence_text_lower = sentence.text.lower()  # Küçük harfe çevir
        for aspect, keywords in aspects.items():
            if any(keyword.lower() in sentence_text_lower for keyword in keywords):
                scores = analyzer.polarity_scores(sentence.text)  # Sentiment analizi yap
                sentiment_scores[aspect].append(scores['compound'])

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

df = df[:5]

results = df["Split_Review"].apply(lambda x: aspect_based_sentiment_analysis(x, aspects))

sentiment_df = pd.DataFrame(results.tolist(), index=df.index)

df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

output_file = "./Data/output/Vader_ABSA.csv"
df_with_sentiment.to_csv(output_file, index=False)
