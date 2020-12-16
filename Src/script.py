import os, pickle, re
import numpy as np
import nltk
import sklearn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

beginning_sentence_indices = []


def clean_sentences(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = [re.sub("[^a-zA-Z0-9]", " ", s) for s in sentences]
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return clean_sentences


# helper function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def to_sentences(text):
    return sent_tokenize(text)


def get_word_embeddings():
    return pickle.load(open('glove.10K.100d.pickle', 'rb'))


def get_sentence_vectors(sentences, word_embeddings):
    sentence_vectors = []
    for s in sentences:
        if len(s) > 0:
            v = [word_embeddings.get(w, np.zeros((100,))) for w in s.split()]
            count = np.sum([emb != np.zeros((100,)) for emb in v])
            v = np.sum(v, axis=0) / (count + 0.0001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors


def partition_sentences(sentence_vectors, sentences):
    beginning_current_section = 0
    similarity_threshold = 0.78
    divided_sentences = []
    for i in range(len(sentence_vectors)):
        sum_vectors = np.zeros((100,))
        for vector in sentence_vectors[beginning_current_section:i]:
            sum_vectors += vector
        avg_sentence = sum_vectors / (len(sentence_vectors[beginning_current_section:i]) + 0.0001)
        if (i - beginning_current_section > 1 and cosine_similarity(sentence_vectors[i].reshape(1, 100),
                                                                    avg_sentence.reshape(1,
                                                                                         100)) < similarity_threshold):
            divided_sentences.append(sentences[beginning_current_section:i])
            beginning_sentence_indices.append(beginning_current_section)
            beginning_current_section = i
        if (i == len(sentence_vectors) - 1):
            divided_sentences.append(sentences[beginning_current_section:i + 1])
            beginning_sentence_indices.append(beginning_current_section)
    return divided_sentences


def format_partitioned_sent(partitioned_sent):
    sentence_groups = []
    for partition in partitioned_sent:
        sentence_group = ""
        for sent in partition:
            sentence_group += sent
        sentence_groups.append(sentence_group)
    return sentence_groups


def get_sentence_beginning_word_indices(sentences):
    sentence_beginning_word_indices = []
    x = 0
    for sentence in sentences:
        sentence_beginning_word_indices.append(x)
        x += len(sentence.split())
    return sentence_beginning_word_indices


def get_section_beginning_timestamps(timestamp_list, sentence_beginning_word_indices):
    section_beginning_timestamps = []
    for i in beginning_sentence_indices:
        section_beginning_timestamps.append(timestamp_list[sentence_beginning_word_indices[i]])
    return section_beginning_timestamps


def format_timestamps(section_beginning_timestamps):
    formatted_timestamps = []
    for timestamp in section_beginning_timestamps:
        total_seconds = int(timestamp * (10 ** -7))
        minutes = int(total_seconds / 60)
        seconds = int(total_seconds % 60)
        formatted_timestamps.append(str(minutes).zfill(2) + ":" + str(seconds).zfill(2))
    return formatted_timestamps


def get_section_topics(formatted_partitioned_sent):
    section_topics = []
    for text in formatted_partitioned_sent:
        section_topic = ""
        words = text.split()
        counter = Counter(words)
        most_occur = counter.most_common(3)
        for pair in most_occur:
            section_topic += pair[0]
            section_topic += " | "
        section_topics.append(section_topic)
    return section_topics


def main(text, timestamp_list):
    sentences = to_sentences(text)
    clean_sent = clean_sentences(sentences)
    word_embeddings = get_word_embeddings()
    sentence_vectors = get_sentence_vectors(clean_sent, word_embeddings)
    partitioned_sent = partition_sentences(sentence_vectors, clean_sent)
    sentence_beginning_word_indices = get_sentence_beginning_word_indices(sentences)
    section_beginning_timestamps = get_section_beginning_timestamps(timestamp_list, sentence_beginning_word_indices)
    formatted_partitioned_sent = format_partitioned_sent(partitioned_sent)
    section_topics = get_section_topics(formatted_partitioned_sent)
    return (section_topics, format_timestamps(section_beginning_timestamps))


if __name__ == "__main__":
    main("text")
