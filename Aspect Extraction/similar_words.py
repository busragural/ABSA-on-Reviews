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



def update_frequencies_with_similar_words(frequency_file, model_w2v, model_fasttext, similarity_threshold=0.8, output_file="./Data/output/updated_frequency_word.csv"):
    word_frequency_df = pd.read_csv(frequency_file)
    
    # İlk 500 kelimeyi al
    top_words = word_frequency_df.head(500)
    
    # Yeni sütunlar ekle
    top_words['Updated_Frequency'] = top_words['Frequency']
    top_words['Similar_Words'] = ""  

    already_processed_similar_words = set()

    for index, row in top_words.iterrows():
        main_word = row['Word']
        main_frequency = row['Frequency']
        
        # Word2Vec ve FastText'ten benzer kelimeleri bul
        similar_words_w2v = [x for x in model_w2v.wv.most_similar(positive=[main_word], topn=100) if x[1] >= similarity_threshold]
        similar_words_ft = [x for x in model_fasttext.wv.most_similar(positive=[main_word], topn=100) if x[1] >= similarity_threshold]
        
        similar_frequencies = 0
        similar_words = []
        
        for similar_word, _ in similar_words_w2v + similar_words_ft:
            if similar_word in already_processed_similar_words:
                continue
            
            if similar_word in top_words['Word'].values:
                continue
            
            if similar_word in word_frequency_df['Word'].values:
                similar_frequency = word_frequency_df.loc[word_frequency_df['Word'] == similar_word, 'Frequency'].values[0]
                similar_frequencies += similar_frequency
                similar_words.append(similar_word)
                already_processed_similar_words.add(similar_word)  
        
        
        top_words.at[index, 'Updated_Frequency'] = main_frequency + similar_frequencies
        top_words.at[index, 'Similar_Words'] = ", ".join(similar_words) if similar_words else "None"
    top_words.to_csv(output_file, index=False)
    print(f"Updated frequencies and similar words saved to {output_file}")


if __name__ == "__main__":

    reviews = []
    for i in reviews_df["Edited_Review"]:
        reviews.append(clean_text(i)) 


    model_w2v = Word2Vec(reviews, vector_size=150, window=10, min_count=2, workers=10, sg=0 )  #sg=0 cbow
    model_w2v.train(reviews, total_examples=len(reviews), epochs=12)


    model_fasttext = FastText(vector_size=170, window=10, min_count=2, workers=10)  # instantiate
    model_fasttext.build_vocab(reviews)
    model_fasttext.train(reviews, total_examples=len(reviews), epochs=10)  # train

    word_vectors = model_fasttext.wv
    word_vectors_w2v = model_w2v.wv


    update_frequencies_with_similar_words("./Data/output/frequency_all_words.csv", model_w2v, model_fasttext)

    
