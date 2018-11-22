import numpy as np

f = open('Project (Application 1) (MetuSabanci Treebank).conll')

SOS = "<s>" #Start of sentence
EOS = "</s>" #End of sentence



def create_n_gram(data, n=2):
    observations = {}
    for sentence in data:
        for i in range(len(sentence)-n+1):
            n_gram_word = ""
            for j in range(n):
                n_gram_word += f'{sentence[i+j][0]} '
            n_gram_word = n_gram_word.strip()
            observations[n_gram_word] = observations.get(n_gram_word, 0) + 1
    total_occurences = np.sum(list(observations.values()))
    for k in observations.keys():
        observations[k] /= total_occurences
    return observations

def get_transition_dict(data):
    transitions = {}
    for sentence in data:
        for i in range(len(sentence)-1):
            transitions[sentence[i][1]] = transitions.get(sentence[i][1], {})
            transitions[sentence[i][1]][sentence[i+1][1]] = transitions[sentence[i][1]].get(sentence[i+1][1], 0) + 1

    for first_tag, second_tags in transitions.items():
        total_occurrence = np.sum(list(second_tags.values()))
        print(first_tag, total_occurrence)
        for k in second_tags.keys():
            second_tags[k] /= total_occurrence
        transitions[first_tag] = second_tags

    return transitions

def viterbi(observation_dict, transition_dict, test):
    pass

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
train_data = data[:int(len(data)*0.9)]
test_data = data[int(len(data)*0.9):]

observations_dict = create_n_gram(train_data, n=2)
transitions_dict = get_transition_dict(train_data)

