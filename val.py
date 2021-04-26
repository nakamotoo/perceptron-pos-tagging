# -*- coding: utf-8 -*-
import numpy as np
import random
from tqdm import tqdm


def create_word_dict(filename):
    # {"word" : (word_id, count)}
    word_count_dict = {"UNK": 100}

    f = open(filename, 'r')

    for line in f.readlines():
        line = line.strip()
        tokens = line.split(' ')

        for index, token in enumerate(tokens):
            if len(token) == 0:
                continue
            if "\\*" in token:
                token = token.replace("\\*", "*")
            if "\\/" in token:
                token = token.replace("\\/", "#")

            word, _ = token.split('/')

            word = word.lower()

            if word not in word_count_dict.keys():
                word_count_dict[word] = 1
            else:
                word_count_dict[word] += 1

    word_dict = {}
    word_id = 0
    for word, count in word_count_dict.items():
        if count <= 8:
            continue
        else:
            word_dict[word] = word_id
            word_id += 1

    return word_dict


class Sentence:
    def __init__(self, labels, word_ids, words, pos_tags, tokens):
        self.labels = labels
        self.word_ids = word_ids
        self.words = words
        self.pos_tags = pos_tags
        self.tokens = tokens
        self.features = None

    def show(self):
        print(self.tokens)
        print(self.words)
        print(self.pos_tags)
        print(self.labels)
        print(self.word_ids)
        print(features.shape)

    def extract_features(self, dim_words):

        if self.features is not None:
            return self.features

        features = []
        one_hot_table = np.identity(dim_words)[self.word_ids]
        prev_one_hot = np.array([0 for i in range(dim_words)])

        for index, word in enumerate(self.words):

            one_hot = one_hot_table[index].tolist()

            feature_dict = {
                'is_first_capital': int(word[0].isupper()),
                'is_first_word': int(index == 0),
                'is_last_word': int(index == len(self.words) - 1),
                'is_numeric': int(word.isdigit()),
                'is_all_capital': int(word.upper() == word),
                'is_end_s' : int(word[-1] == 's'),
                'is_end_ing' : int(word[-3:] == 'ing') if len(word) > 3 else 0,
                'is_end_tion' : int(word[-4:] == 'tion') if len(word) > 4 else 0,
                'is_end_ed' : int(word[-2:] == 'ed') if len(word) > 3 else 0,
                'is_end_y' : int(word[-1:] == 'y')            }

            feature = one_hot + list(prev_one_hot) + list(feature_dict.values())
            features.append(feature)
            prev_one_hot = one_hot

        features = np.array(features)
        self.features = features
        return features


def read_samples(filename):
    f = open(filename, 'r')
    data = []

    for line in tqdm(f.readlines()):
        line = line.strip()
        tokens = line.split(' ')

        word_ids = []
        labels = []
        words = []
        pos_tags = []
        features = []

        len_sentence = len(tokens)

        prev_word_id = -1

        for index, token in enumerate(tokens):
            if len(token) == 0:
                continue
            if "\\*" in token:
                token = token.replace("\\*", "*")
            if "\\/" in token:
                token = token.replace("\\/", "#")

            word, pos = token.split('/')
            words.append(word)

            pos_tags.append(pos)

            if pos not in pos_dict.keys():
                pos_dict[pos] = len(pos_dict)

            pos_id = pos_dict[pos]
            labels.append(pos_id)

            lower_word = word.lower()
            if lower_word not in word_dict.keys():
                lower_word = "UNK"

            word_id = word_dict[lower_word]
            word_ids.append(word_id)
            prev_word_id = word_id

        sentence = Sentence(labels, word_ids, words, pos_tags, tokens)
        data.append(sentence)

    return data


def classify(sentence, features, dim_pos):
    predicted_pos_ids = []
    prev_predicted_id = 0

    features_aug = []
    for ind, word_id in enumerate(sentence.word_ids):
        feature_vec = features[ind]
        prev_predicted_pos_vec = np.identity(dim_pos)[prev_predicted_id]

        feature_vec = np.append(feature_vec, prev_predicted_pos_vec)

        max_score = -np.inf
        max_i = -1

        for i in range(len(weight_vectors)):
            score = np.dot(weight_vectors[i], feature_vec)
            if score > max_score:
                max_score = score
                max_i = i

        predicted_pos_ids.append(max_i)
        prev_predicted_pos_id = max_i
        features_aug.append(feature_vec)

    return predicted_pos_ids, features_aug


pos_dict = {}
word_dict = create_word_dict("data/train.pos")


training_data = []
valid_data = []
training_data = read_samples("data/train.pos")
print(len(pos_dict), len(word_dict))
val_data = read_samples("data/val.pos")
print(len(pos_dict), len(word_dict))

dim_words = len(word_dict)
dim_pos = len(pos_dict)

# initialization
fe = training_data[0].extract_features(dim_words)
feature_num = fe.shape[1] + dim_pos
weight_vectors = np.random.uniform(
    low=-0.08, high=0.08, size=(len(pos_dict), feature_num)).astype('float32')
u_vectors = np.zeros(shape=(len(pos_dict), feature_num)).astype('float32')


weight_vectors = np.load("w_0.npy")
total = 0
correct = 0
for sentence in tqdm(val_data):
    features = sentence.extract_features(dim_words)
    predicted_pos_ids, _ = classify(sentence, features, dim_pos)

    for i in range(len(sentence.labels)):
        label, y = sentence.labels[i], predicted_pos_ids[i]
        if (y == label):
            correct += 1
        total += 1
print("VAL W", "num_updates = ", num_updates,
        "accuracy = ", correct / total * 100)

weight_vectors = np.load("v_0.npy")
total = 0
correct = 0
for sentence in tqdm(val_data):
    features = sentence.extract_features(dim_words)
    predicted_pos_ids, _ = classify(sentence, features, dim_pos)

    for i in range(len(sentence.labels)):
        label, y = sentence.labels[i], predicted_pos_ids[i]
        if (y == label):
            correct += 1
        total += 1
print("VAL V", "num_updates = ", num_updates,
        "accuracy = ", correct / total * 100)


