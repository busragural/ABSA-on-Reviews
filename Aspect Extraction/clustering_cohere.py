from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import pandas as pd 
import cohere
import numpy as np


key = "FlG7ICoQ2QjBXhRsyCpFA1TocJhvPmxSSvy3wqEs"

df = pd.read_csv("/Users/busragural/Desktop/4.1/ABSA/Data/output/frequency_all_words.csv")
df_filtered = df[df["Frequency"] > 1]
words = df_filtered["Word"].dropna().unique().tolist()

co = cohere.Client(key)
response = co.embed(texts=words, model="embed-multilingual-v3.0", input_type="clustering")
word_vectors = np.array(response.embeddings)


# Hiyerarşik kümeleme kullanarak gruplama
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
clusters = clustering.fit_predict(word_vectors)

grouped_words = {}
for word, cluster in zip(words, clusters):
    grouped_words.setdefault(cluster, []).append(word)

for cluster_id, word_list in grouped_words.items():
    print(f"Küme {cluster_id}: {word_list}")


clustered_df = pd.DataFrame(
    [(cluster, ", ".join(words)) for cluster, words in grouped_words.items()], 
    columns=["Cluster", "Words"]
)

file_path = "./Data/output/clustered_words.csv"
clustered_df.to_csv(file_path, index=False)