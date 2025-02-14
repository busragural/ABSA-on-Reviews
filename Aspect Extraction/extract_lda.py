import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')  # Lemmatization
nltk.download('omw-1.4')
nltk.download('stopwords')
stop_words = stopwords.words('english')

df = pd.read_csv('/Users/busragural/Desktop/4.1/Bitirme/gemini_ile_preprocess/preprocessingWithGemini.csv')

df['Split_Review'] = df['Split_Review'].fillna('')

comments = df['Split_Review'].tolist()

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower() 
    words = word_tokenize(text) 
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

cleaned_comments = [clean_text(comment) for comment in comments]

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=300, stop_words=stop_words)  
tfidf = tfidf_vectorizer.fit_transform(cleaned_comments)

num_topics = 5  # number of aspects
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(tfidf)

def display_topics(model, feature_names, num_top_words):
    topic_word_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[:-num_top_words - 1:-1]
        
        topic_words = []
        for i in top_word_indices:
            word = feature_names[i]
            tokens = word_tokenize(word)
            pos_tags = nltk.pos_tag(tokens)
            if all(tag.startswith('NN') for word, tag in pos_tags): #only nouns
                topic_words.append(word)

        topic_word_dict[f"Topic {topic_idx + 1}"] = topic_words
        print(f"Topic {topic_idx + 1}:")
        print(" ".join(topic_words))
    return topic_word_dict

num_top_words = 20  # Her aspect için kaç kelime görüntülenecek
topic_word_dict = display_topics(lda, tfidf_vectorizer.get_feature_names_out(), num_top_words)

all_aspects = []
for topic, words in topic_word_dict.items():
    all_aspects.extend(words)

all_aspects = list(set(all_aspects))

print("\nAspects:")
print(all_aspects)

#['ambience', 'experience', 'restaurant', 'food', 'chicken', 'service', 'taste', 'visit', 'order', 'staff', 'time', 'place', 'try']