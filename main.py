import numpy as np

f = open('Project (Application 1) (MetuSabanci Treebank).conll', encoding='utf-8')

SOS = "<s>"  # Start of sentence
EOS = "</s>"  # End of sentence


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


def viterbi(observation_dict, transition_dict, sentence):
    states = list(transition_dict.keys())
    viterbi = np.zeros((len(states) + 1, len(sentence) - 1))
    backpointer = np.zeros((len(states) + 1, len(sentence) - 1))

    ### Initialization step
    for s in range(1, len(states)):
        current_tag = states[s]
        viterbi[s, 1] = transition_dict[SOS].get(current_tag, 0) * observation_dict[sentence[1][0]].get(current_tag, 0)
        backpointer[s, 1] = 0

    ### Recursion Step
    for t in range(2, len(sentence) - 1):
        for s in range(1, len(states)):
            current_tag = states[s]
            max_path = 0;
            max_state = 0;
            for i in range(1, len(states)):
                previous_tag = states[i]
                path_prob = viterbi[i, t - 1] * transition_dict[previous_tag].get(current_tag, 0) * observation_dict[
                    sentence[t][0]].get(current_tag, 0)
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
        if expected_tags[i] != predicted_tags[i]:
            return False
    return True


### Read Data
data = []
sentence = []
for line in f:
    columns = line.split()
    if len(columns) == 10 and columns[1] != "_":
        ### Handle the wrong annotation
        if columns[3] == "satın":
            columns[3] = "Noun"
        sentence.append([columns[1], columns[3]])
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

sum = 0
for sentence in train_data:
    backtrack = viterbi(observations_dict, transitions_dict, sentence)
    result = get_backtrack(backtrack, tags)
    if check_sentence(get_tags(sentence), result):
        sum += 1

print(f'Accuracy: {sum/len(train_data)}')


random_sentence = train_data[np.random.randint(len(train_data))]
print(random_sentence)
backtrack = viterbi(observations_dict, transitions_dict, random_sentence)
result = get_backtrack(backtrack, tags)
