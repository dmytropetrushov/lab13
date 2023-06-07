import math
from collections import Counter


def compute_idf(word, corpus):
    return math.log10(len(corpus) / sum(1.0 for doc in corpus if word in doc))


def compute_tfidf(corpus):
    document_list = []  # List of dictionaries for each document
    for text in corpus:
        tf_idf_dictionary = {}  # Dictionary for each document in the corpus
        computer_tf = Counter(text)  # Counter dictionary for each text, representing TF values
        for word in computer_tf:
            tf_idf_dictionary[word] = computer_tf[word] * compute_idf(word, corpus)
        document_list.append(tf_idf_dictionary)
    return document_list


# Read the contents of the files
file_names = ['file_lab13_1.txt', 'file_lab13_2.txt', 'file_lab13_3.txt']
corpus = []
for file_name in file_names:
    with open(file_name, 'r') as file:
        text = file.read().split()  # Split the text into a list of words
        corpus.append(text)

# Compute TF-IDF weights for the corpus
tfidf_weights = compute_tfidf(corpus)

# Combine TF-IDF weights for all documents
combined_tfidf = Counter()
for tfidf in tfidf_weights:
    combined_tfidf.update(tfidf)

# Get the 20 terms with the highest TF-IDF weights
top_terms = combined_tfidf.most_common(20)

# Print the top terms
for term, weight in top_terms:
    print(f'Term: {term}\tTF-IDF Weight: {weight}')
