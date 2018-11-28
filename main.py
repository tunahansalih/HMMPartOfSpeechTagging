#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random as rn
import re
import matplotlib.pyplot as plt
import pandas
from pandas_ml import ConfusionMatrix

pandas.set_option('display.max_columns', None)

f = open('Project (Application 1) (MetuSabanci Treebank).conll', encoding='utf-8')

SOS = "<s>"  # Start of sentence
EOS = "</s>"  # End of sentence
observations_dict = {}
transitions_dict = {}

rules = [r'.*(lar|ler)[aeıi]?$', r'.*(ti|tı|dı|di|du|dü|tu|tü)$', r'.*(yor).*$', r'.*(cak|cek).*$', r'^[A-ZIÇŞÜÖ].*']
# rules = []
unknown_word_rules = {}
general_dist = {}


def get_observation_dict(data):
    print('Creating Observation Dictionary')
    observations = {}
    for sentence in data:
        for word in sentence:
            observations[word[0]] = observations.get(word[0], {})
            observations[word[0]][word[1]] = observations[word[0]].get(word[1], 0) + 1
    for word, v in observations.items():
        total_occurence = np.sum(list(observations[word].values()))
        for tag, v1 in observations[word].items():
            observations[word][tag] /= total_occurence
    return observations


def get_transition_dict(data):
    print('Creating Transition Dictionary')
    transitions = {}
    for sentence in data:
        for i in range(len(sentence) - 1):
            transitions[sentence[i][1]] = transitions.get(sentence[i][1], {})
            transitions[sentence[i][1]][sentence[i + 1][1]] = transitions[sentence[i][1]].get(sentence[i + 1][1], 0) + 1

    for first_tag, second_tags in transitions.items():
        total_occurrence = np.sum(list(second_tags.values()))
        for k in second_tags.keys():
            second_tags[k] /= total_occurrence
        transitions[first_tag] = second_tags

    return transitions


def fill_general_distribution(data):
    words = {}
    tags = {}
    dist = {}
    for sentence in data:
        for word in sentence:
            if word[1] != SOS and word[1] != EOS:
                words[word[0]] = words.get(word[0], 0) + 1
                tags[word[0]] = word[1]

    for word, count in words.items():
        if count == 1:
            dist[tags[word]] = dist.get(tags[word], 0) + 1

    sum = np.sum(list(dist.values()))
    dist = {k: v / sum for k, v in dist.items()}
    return dist


def viterbi(observation_dict, transition_dict, sentence):
    states = list(transition_dict.keys())
    viterbi = np.zeros((len(states) + 1, len(sentence) - 1))
    backpointer = np.zeros((len(states) + 1, len(sentence) - 1))

    ### Initialization step
    for s in range(1, len(states)):
        current_tag = states[s]
        viterbi[s, 1] = transition_dict[SOS].get(current_tag, 0) * observation_dict.get(sentence[1][0], unknown_handle(
            sentence[1][0])).get(current_tag, 0)
        backpointer[s, 1] = 0

    ### Recursion Step
    for t in range(2, len(sentence) - 1):
        for s in range(1, len(states)):
            current_tag = states[s]
            max_path = 0;
            max_state = 0;
            for i in range(1, len(states)):
                previous_tag = states[i]
                path_prob = viterbi[i, t - 1] * transition_dict[previous_tag].get(current_tag,
                                                                                  0) * observation_dict.get(
                    sentence[t][0], unknown_handle(sentence[t][0])).get(current_tag, 0)
                if path_prob > max_path:
                    max_path = path_prob
                    max_state = i
            viterbi[s, t] = max_path
            backpointer[s, t] = max_state

    # Termination Step
    max_final = 0;
    max_final_point = 0;
    for s in range(1, len(states)):
        current_tag = states[s]
        curr_prob = viterbi[s, len(sentence) - 2] * transition_dict[current_tag].get(EOS, 0)
        if (curr_prob > max_final):
            max_final = curr_prob
            max_final_point = s

    viterbi[len(states), len(sentence) - 2] = max_final
    backpointer[len(states), len(sentence) - 2] = max_final_point

    return backpointer


def unknown_handle(word):
    for pattern, dist in unknown_word_rules.items():
        match = re.match(pattern, word)
        if match:
            return dist

    return general_dist


def fill_rule_distributions(rules):
    for rule in rules:
        dist_dict = {}
        for sentence in train_data:
            for word in sentence:
                match = re.search(rule, word[0])
                if match:
                    dist_dict[word[1]] = dist_dict.get(word[1], 0) + 1
        sum = np.sum(list(dist_dict.values()))
        dist_dict = {k: v / sum for k, v in dist_dict.items()}
        unknown_word_rules[rule] = dist_dict


def get_backtrack(backtrack_matrice, tags_in_corpus):
    tags = np.array([])
    current_point = int(backtrack_matrice[backtrack_matrice.shape[0] - 1, backtrack_matrice.shape[1] - 1])
    tags = np.insert(tags, 0, current_point)
    for i in range(backtrack_matrice.shape[1] - 1):
        current_point = int(backtrack_matrice[current_point, backtrack_matrice.shape[1] - i - 1])
        tags = np.insert(tags, 0, current_point)

    result = []
    for tag in tags.astype(np.int32):
        result.append(tags_in_corpus[tag])
    result.append(EOS)

    return result


def get_tags(sentence):
    tags = []
    for word in sentence:
        tags.append(word[1])
    return tags


def check_sentence(expected_tags, predicted_tags):
    for i in range(len(predicted_tags)):
        all_expected_tags.append(expected_tags[i])
        all_predicted_tags.append(predicted_tags[i])
        if expected_tags[i] != predicted_tags[i]:
            return False
    return True


def get_word_based_accuracy(expected_tags, predicted_tags):
    sum = 0
    for e, p in zip(expected_tags, predicted_tags):
        if e == p:
            sum += 1
    return sum / len(expected_tags)


### Read Data
data = []
sentence = []
for line in f:
    columns = line.split()
    if len(columns) == 10 and columns[1] != "_":
        ### Handle the wrong annotation
        if columns[3] == "satın":
            columns[3] = "Noun"
        sentence.append([columns[1].lower(), columns[3]])
        # sentence.append([columns[1], columns[3]])
    if len(columns) == 0:
        data.append(sentence)
        sentence = []

### Insert <s> at the head and </s> to the end
for sentence in data:
    sentence.insert(0, [SOS, SOS])
    sentence.append([EOS, EOS])

### Create train and test data
np.random.shuffle(data)
train_data = data[:int(len(data) * 0.9)]
test_data = data[int(len(data) * 0.9):]

### Train Data
observations_dict = get_observation_dict(train_data)
transitions_dict = get_transition_dict(train_data)
tags = list(transitions_dict.keys())
tags.append(EOS)

general_dist = fill_general_distribution(train_data)
fill_rule_distributions(rules)

all_expected_tags = []
all_predicted_tags = []

print(general_dist)

sum = 0
for sentence in test_data:
    backtrack = viterbi(observations_dict, transitions_dict, sentence)
    result = get_backtrack(backtrack, tags)
    if check_sentence(get_tags(sentence), result):
        sum += 1

print(f'Sentece-Based Accuracy: {sum/len(test_data)}')

print(f'Word-based Accuracy: {get_word_based_accuracy(all_expected_tags, all_predicted_tags)}')

cm = ConfusionMatrix(all_expected_tags, all_predicted_tags)
cm.plot()
plt.show()

stats = cm.stats()

f = open('stats_lower.txt', 'w')
f.write(f'Overall Statistics:\n')
for k, v in stats['overall'].items():
    f.write(f'{k}: {v}\n')

for k, v in stats['class'].items():
    f.write(f'{k}: {v}\n')

f.write(str(stats['overall']))
f.write(str(stats['class']))

cm = ConfusionMatrix(all_expected_tags, all_predicted_tags)
cm.plot(normalized=True)
plt.show()
cm.print_stats()
f.close()

for k, v in unknown_word_rules.items():
    plt.bar(list(v.keys()), list(v.values()))
    plt.title(label=f'Regex Rule: {k}')
    plt.show()

plt.bar(list(general_dist.keys()), list(general_dist.values()))
plt.title(label='Hapax Legomenon Distribution')
plt.show()
