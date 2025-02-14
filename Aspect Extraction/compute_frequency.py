import pandas as pd 
from collections import Counter

import nltk 
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, re, word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger')


def count_frequency(reviews):
    all_words = [word for review in reviews for word in review]  
    frequency = Counter(all_words)  
    
    word_frequency_df = pd.DataFrame(frequency.items(), columns=['Word', 'Frequency'])
    word_frequency_df = word_frequency_df.sort_values(by='Frequency', ascending=False) 

    word_frequency_df.to_csv("./Data/output/frequency_all_words.csv", index=False)
    print("Word frequency saved.")

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [x for x in text if x not in stop_words]
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
                             
    text = [t for t in text if len(t) > 1]     
    return (text)


df = pd.read_csv("./Data/output/preprocessing_base.csv")
reviews_df = df[['Edited_Review']]  

reviews = []
for i in reviews_df["Edited_Review"]:
    reviews.append(clean_text(i)) 

count_frequency(reviews)

