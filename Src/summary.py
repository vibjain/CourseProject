import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')


def extract_word_vectors(word_embeddings):
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def main():
    with open('satya_speech_ready2020.txt', 'r') as file:
        text = file.read()
        sentences = sent_tokenize(text)

        word_embeddings = {}
        extract_word_vectors(word_embeddings)

        # remove punctuations, numbers and special characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]

        # remove stopwords from the sentences
        clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

        sentence_vectors = []

        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((100,))

            sentence_vectors.append(v)

        num_sections = 5
        n = len(sentence_vectors) // num_sections
        section_vectors_lst = [sentence_vectors[i * n:(i + 1) * n] for i in range((len(sentence_vectors) + n - 1) // n)]
        section_sentences_lst = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n - 1) // n)]

        print("len", len(section_vectors_lst))

        for i in range(len(section_vectors_lst)):
            # similarity matrix
            sent_lst = section_sentences_lst[i]
            sent_vector_lst = section_vectors_lst[i]

            sim_mat = np.zeros([len(sent_lst), len(sent_lst)])

            for i in range(len(sent_lst)):
                for j in range(len(sent_lst)):
                    if i != j:
                        sim_mat[i][j] = \
                        cosine_similarity(sent_vector_lst[i].reshape(1, 100), sent_vector_lst[j].reshape(1, 100))[0, 0]

            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)

            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sent_lst)), reverse=True)

            # Extract top sentence as summary
            for i in range(1):
                print(ranked_sentences[i][1])


if __name__ == "__main__":
    main()
