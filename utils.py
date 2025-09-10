from collections import Counter
from nltk import ngrams
import math

def calculate_ngram_weights(df_info, topic_model):
    uni = Counter()
    total_uni = 0
    for toks in df_info['tokens']:
        uni.update(toks)
        total_uni += len(toks)

    ngram_freqs = {}
    topic_ids = [tid for tid in topic_model.get_topics().keys() if tid != -1]
    max_tid = max(topic_ids) if topic_ids else 0

    for tid in topic_ids:
        docs = df_info[df_info['Topic'] == tid]['tokens'].tolist()
        tf = Counter()
        for toks in docs:
            for gram in ngrams(toks, 2):
                tf[gram] += 1
        factor = (max_tid - tid + 1)
        for gram, cnt in tf.items():
            if cnt < 3:
                continue
            probs = [uni[w] / total_uni for w in gram]
            p_joint = cnt / total_uni
            pmi = math.log(p_joint / math.prod(probs), 2) if all(probs) else 0
            if pmi < 1.0:
                continue
            label = "‌".join(gram)
            weight = cnt * factor * pmi
            ngram_freqs[label] = ngram_freqs.get(label, 0) + weight
    return ngram_freqs
