import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import random
import os

def load_imdb_sentiment_analysis_dataset(data_path, seed = 123):
    
    imdb_data_path = os.path.join(data_path, 'aclImdb')
    
    # Load the traning data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with(open(os.path.join(train_path, fname), encoding='utf-8')) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)
    
    # Load the validation data
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with(open(os.path.join(test_path, fname), encoding='utf-8')) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, train_labels), (test_texts, test_labels))

def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def plot_sample_length_distribuition(sample_texts):
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Lenght of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribuition')
    plt.show()

def get_num_classes(labels):
    num_classes = max(labels) + 1
    missin_classes = [i for i in range(num_classes) if i not in labels]

    if len(missin_classes):
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}. Please make sure you have '
                         'at least one sample for every label value '
                         'in the range(0, {max_class})'.format(
                            missing_classes=missing_classes,
                            max_class=num_classes - 1))
    
    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    
    return num_classes

def ngram_vectorize(train_texts, train_labels, val_texts):
    # Vectorization parameters
    NGRAM_RANGE = (1, 2) # Range (inclusive) of n-gram sizes for tokenizing text
    
    TOP_K = 20000 # Limit on the number of features
    
    TOKEN_MODE = 'word' # Wheter text should be split intp word or character n-grams

    MIN_DOCUMENT_FREQUENCY = 2 # Minimum document/corpus frequency below which a token will ne discared 

    kwargs = {
        'ngram_range': NGRAM_RANGE,
        'dtype': 'int32',
        'strip_accents':'unicode',
        'decode_error':'replace',
        'analyzer':TOKEN_MODE,
        'min_df': MIN_DOCUMENT_FREQUENCY
    }

    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training and vectorize training texts
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')

    return x_train, x_val

train_data, test_data = load_imdb_sentiment_analysis_dataset('C:\\Users\\lucas\\Desktop')

x_train, x_val = ngram_vectorize(train_data[0], train_data[1], test_data[0])
print(x_train[0:5].toarray()cd)

# num_words_per_sample = get_num_words_per_sample(train_data[0])
# print(num_words_per_sample)
# print(get_num_classes(train_data[1]))
# plot_sample_length_distribuition(train_data[0])