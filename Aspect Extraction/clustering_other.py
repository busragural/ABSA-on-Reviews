import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("/Users/busragural/Desktop/4.1/ABSA/Data/output/frequency_all_words.csv")

#df_filtered = df[df["Frequency"] > 20]

words = df["Word"].dropna().unique().tolist()

#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("all-mpnet-base-v2")

word_vectors = model.encode(words)

clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
clusters = clustering.fit_predict(word_vectors)

grouped_words = {}
for word, cluster in zip(words, clusters):
    grouped_words.setdefault(cluster, []).append(word)

for cluster_id, word_list in grouped_words.items():
    print(f"Küme {cluster_id}: {word_list}")
