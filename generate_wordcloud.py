import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from umap import UMAP
from bertopic import BERTopic
from embedder import ParsBERTEmbedder
from preprocess import safe_preprocess
from utils import calculate_ngram_weights
import os


def generate_wordcloud(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["txtContent"])

    texts_raw = df["txtContent"].tolist()
    tokens_list = [safe_preprocess(txt) for txt in texts_raw]
    texts = [" ".join(tokens) for tokens in tokens_list]

    if not texts:
        print("⚠️ No valid texts for embedding.")
        return

    print("🔎 Embedding documents...")
    embedder = ParsBERTEmbedder()
    embs = embedder.embed(texts)

    print("🧠 Training BERTopic model...")
    umap_model = UMAP(
        n_neighbors=max(1, min(15, len(texts) - 1)),
        n_components=2,
        min_dist=0.1,
        metric="cosine",
    )
    topic_model = BERTopic(
        language="persian",
        embedding_model=embedder,
        umap_model=umap_model,
        calculate_probabilities=True,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(texts, embs)
    df_info = topic_model.get_document_info(texts)
    df_info["tokens"] = tokens_list

    print("📊 Calculating n-gram weights...")
    ngram_freqs = calculate_ngram_weights(df_info, topic_model)

    if ngram_freqs:
        print("🎨 Generating word cloud...")
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="tab20",
            max_words=200,
            min_font_size=10,
            font_path="Vazir-Bold.ttf",
        )
        wc.generate_from_frequencies(ngram_freqs)
        os.makedirs("wordclouds", exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        filename = "wordclouds/wordcloud_all_data.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved {filename}")
    else:
        print("⚠️ No valid n-grams found.")
