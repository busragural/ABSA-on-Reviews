import pandas as pd
from collections import Counter
import nltk 
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, re, word_tokenize
from gensim.models import FastText, Word2Vec
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger')


df = pd.read_csv("./Data/output/preprocessing_base.csv")
reviews_df = df[['Edited_Review']]  

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

def word2vectfonc(kdizi, k):
    dizi_w2v = [list(x) for x in model_w2v.wv.most_similar(positive=kdizi, topn=20)]      

    filtered_w2v = [list(x) for x in dizi_w2v if x[1] >= 0.8]
    
    d_w2v=[]

    for i in range(len(filtered_w2v)):
        d_w2v.append(filtered_w2v[i][0])  
        filtered_w2v[i][1] = format(filtered_w2v[i][1], '.2f')  

    print(f"{k}. Most similar to (w2v) {kdizi} with similarity >= 0.8:", filtered_w2v)


def update_frequencies_with_similar_words(frequency_file, model_w2v, similarity_threshold=0.5, output_file="./Data/output/updated_frequency_words_w2v.csv"):
    word_frequency_df = pd.read_csv(frequency_file)
    
    # İlk 500 kelimeyi al
    top_words = word_frequency_df.head(500)
    
    # Yeni sütunlar ekle
    top_words['Updated_Frequency_w2v'] = top_words['Frequency']
    top_words['Similar_Words_w2v'] = ""  

    already_processed_similar_words = set()

    for index, row in top_words.iterrows():
        main_word = row['Word']
        main_frequency = row['Frequency']
        
        # Word2Vec ve FastText'ten benzer kelimeleri bul
        similar_words_w2v = [x for x in model_w2v.wv.most_similar(positive=[main_word], topn=100) if x[1] >= similarity_threshold]
        
        similar_frequencies = 0
        similar_words = []
        
        for similar_word, _ in similar_words_w2v:
            if similar_word in already_processed_similar_words:
                continue
            
            if similar_word in top_words['Word'].values:
                continue
            
            if similar_word in word_frequency_df['Word'].values:
                similar_frequency = word_frequency_df.loc[word_frequency_df['Word'] == similar_word, 'Frequency'].values[0]
                similar_frequencies += similar_frequency
                similar_words.append(similar_word)
                already_processed_similar_words.add(similar_word)  
        
        
        top_words.at[index, 'Updated_Frequency_w2v'] = main_frequency + similar_frequencies
        top_words.at[index, 'Similar_Words_w2v'] = ", ".join(similar_words) if similar_words else "None"
    top_words.to_csv(output_file, index=False)
    print(f"Updated frequencies and similar words saved to {output_file}")


if __name__ == "__main__":

    reviews = []
    for i in reviews_df["Edited_Review"]:
        reviews.append(clean_text(i)) 


    model_w2v = Word2Vec(reviews, vector_size=150, window=10, min_count=2, workers=10, sg=0 )  #sg=0 cbow
    model_w2v.train(reviews, total_examples=len(reviews), epochs=12)

    word_vectors_w2v = model_w2v.wv


    update_frequencies_with_similar_words("./Data/output/frequency_all_words.csv", model_w2v)

    
