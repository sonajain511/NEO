import gensim
import scipy
from scipy.spatial import distance
import pandas
# from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import statistics as s
from matplotlib import ticker
import re
import math
import os

# need to research which model would work best
# sbert_model = SentenceTransformer('nli-bert-large')

pretrained_embeddings_path = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)


# cosine similarity
# instead do average across 5 sentences pairs (1,2), (1,3), (1,4)...(4,5)
# take out first four
# (2,3), (2,4), (2,5), (2,6) <-- 4th index
# (3,4), (3,5), (3,6) <-- 3rd index
# (4,5), (4,6) <-- 2nd index
# (5,6) <-- 1st index
# take similarities scores array (multiply each by last sentence similarity)
# take utterances array (past 5 utterances)
# mark first occurence only

# getting new set of sentence pairs
def retrieve_next_sentence_array(iteration, index, array):
    count = index
    original_index = index
    reverse_count = 1
    while count != 1:
        print("Array:", array)
        array.insert(index, (array[index - 1][0], array[index - 1][1] + 1))
        index = index + original_index - (reverse_count - 1)  # attempt to adjust to altering window sizes
        iteration = iteration + 1
        count = count - 1
        reverse_count = reverse_count + 1
    array.append((array[len(array) - 1][0], array[len(array) - 1][1] + 1))
    array.append((array[len(array) - 1][0] + 1, array[len(array) - 1][1]))
    print("Window Set:", array)
    return array


# calculate similarities between two set of sentences within our window set
def similarity_calculations(array, line):
    sim = 0
    for index in range(len(array)):
        query = line[array[index][0]]
        # query_vec = sbert_model.encode([query])[0]
        query_vec = word2vec[query]
        sim = sim + (1 - scipy.spatial.distance.cosine(query_vec, word2vec[line[array[index][1]]]))
    return sim / len(array)


def generate_graph_w2v_ave_cos_sim(line, filename):
    # set window size
    print("Start")
    window = 5
    sentence_pairs_array = []
    average_similarity_array = []

    iteration = 0
    iteration_array = []

    phase_0_words = ["Google Docs, PDF, hear, shift key, hello, EEG"]
    phase_0_words_sep = ["Google Docs", "PDF", "hear", "shift key", "hello", "EEG"]

    phase_1_words = [
        "Aircraft, helicopter, boats, airports, army base, enterprise, fog, number of people, personnel, speed, rebels, water, shore, five miles, sound, parachuting, paratroopers, tide, sharks, coral reef, village, refuel, arrive"]
    phase_1_words_sep = ["Aircraft", "helicopter", "boats", "airports", "army base", "enterprise", "fog",
                         "number of people", "personnel", "speed", "rebels", "water", "shore", "five miles", "sound",
                         "parachuting", "paratroopers", "tide", "sharks", "coral reef", "village", "refuel", "arrive"]

    phase_2_words = [
        "Ground transportation, cars, trucks, donkeys, rebels, volcanos, personnel, injuries, medical care, church, extract, evacuate, route, rebels, military, village, diabetic, paved road"]
    phase_2_words_sep = ["Ground transportation", "cars", "trucks", "donkeys", "rebels", "volcanos", "personnel",
                         "injuries", "medical care", "church", "extract", "evacuate", "route", "rebels", "military",
                         "village", "diabetic", "paved road"]

    phase_3_words = [
        "Army base, aircraft carrier, enterprise, refuel, return, dirt road, trucks, walk, back, leave, time"]
    phase_3_words_sep = ["Army base", "aircraft carrier", "enterprise", "refuel", "return", "dirt road", "trucks",
                         "walk", "back", "leave", "time"]

    '''
    phase_0_sentence = sbert_model.encode(phase_0_words)[0]
    phase_1_sentence = sbert_model.encode(phase_1_words)[0]
    phase_2_sentence = sbert_model.encode(phase_2_words)[0]
    phase_3_sentence = sbert_model.encode(phase_3_words)[0]
    '''

    phase_0_sentence = word2vec[phase_0_words]
    phase_1_sentence = word2vec[phase_1_words]
    phase_2_sentence = word2vec[phase_2_words]
    phase_3_sentence = word2vec[phase_3_words]



    phase_0_array = []
    phase_1_array = []
    phase_2_array = []
    phase_3_array = []

    '''
    phase_0 = False
    phase_1 = False
    phase_2 = False
    phase_3 = False
    '''

    phases_similarity_array = []
    window_phases_array = []

    matching_coordinates = []
    y_bert_coordinates = []
    y_human_coordinates = []
    y_human_coordinates_mod = []

    y_bert = []
    bert_iteration_array = []

    y_human = []
    human_iteration_array = []

    y_matching_coordinates = []
    matching_iteration_array = []

    # after initial stages gbthnyjmkmbdgbt
    is_zero = True

    # human_phases_array = []
    # print(human_phases_array)

    print("Length of line:", len(line))
    while iteration < len(line) - window + 1 - 2:  # don't evaluate last line --> extra - 1
        # print("Line 1 :", line[iteration + 1])
        # print("Line 2 :", line[iteration + 2])

        new_line = line[iteration + 1] + line[iteration + 2] + \
                   line[iteration + 3] + line[iteration + 4] + line[iteration + 5]
        # line[iteration + 5] + line[iteration + 6] + \
        # line[iteration + 7] + line[iteration + 8] + \
        # line[iteration + 9] + line[iteration + 10]
        if window > 2:
            if iteration == 0:

                for i in range(1, window + 1):
                    # human_phases_array.append(phases_by_human[i])
                    for j in range(1, window + 1):
                        if i < j:
                            sentence_pairs_array.append((i, j))

                if is_zero:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_0_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, word2vec[new_line])]
                else:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, word2vec[new_line])]

                # print("Sentence Pairs Array:", sentence_pairs_array)
                average_similarity_array.append(similarity_calculations(sentence_pairs_array, line))
            else:
                # print("Human_Phases_Array:", human_phases_array)
                # human_phases_array.pop(0)
                # human_phases_array.append(phases_by_human[iteration + window + 1])
                print("Iteration + window + 1:", iteration + window + 1)

                for i in range(0, window - 1):
                    sentence_pairs_array.pop(0)
                # originally iteration
                sentence_pairs_array.append(
                    retrieve_next_sentence_array(iteration + 1, window - 2, sentence_pairs_array))
                # likes to add [...]
                sentence_pairs_array.pop()
                average_similarity_array.append(similarity_calculations(sentence_pairs_array, line))

                # get rid of first line of matrix, since no longer relevant
                # phases_similarity_array.pop(0)
                if is_zero:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_0_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, word2vec[new_line])]
                else:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, word2vec[new_line])]

                # window_phases_array.pop(0)
                # window_phases_array.append(phases_similarity_matrix[len(phases_similarity_matrix) - 1].index(max(phases_similarity_matrix[len(phases_similarity_matrix) - 1])))

            '''
            contained_phases = set(human_phases_array) # gets unique values
            contained_phases = list(contained_phases) # converts back to list

            if len(contained_phases) == 1:
                if contained_phases[0] != 0:
                    y_human_coordinates.append(contained_phases[0])
                else:
                    if y_human_coordinates == [] or y_human_coordinates.count(0) == iteration + 1:
                        y_human_coordinates.append(0)
                    else:
                        y_human_coordinates.append(y_human_coordinates[len(y_human_coordinates) - 1])
            else:
                if iteration != 0:
                    y_human_coordinates.append(y_human_coordinates[len(y_human_coordinates) - 1])
                else:
                    y_human_coordinates.append(0)
            '''

        else:
            print("Iteration:", line[iteration])
            query_prev_sent = word2vec[line[iteration + 1]]
            query_curr_sent = word2vec[line[iteration + 2]]
            average_similarity_array.append(1 - scipy.spatial.distance.cosine(query_prev_sent, query_curr_sent))
            # modify this to incorporate phases_similarity_array if end up testing window = 2

            if is_zero:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_0_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, word2vec[new_line])]
            else:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, word2vec[new_line]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, word2vec[new_line])]

        print("Phases Similarity Array:", phases_similarity_array)
        print("Maximum of Phases Similarity Array:", max(phases_similarity_array))
        print("Index of Maximum of Phases Similarity Array:",
              phases_similarity_array.index(max(phases_similarity_array)))

        if is_zero:
            if phases_similarity_array.index(max(phases_similarity_array)) == 0:
                phase_0_array.append(iteration)
                y_bert_coordinates.append(0)
            elif phases_similarity_array.index(max(phases_similarity_array)) == 1:
                phase_1_array.append(iteration)
                y_bert_coordinates.append(1)
                is_zero = False
            elif phases_similarity_array.index(max(phases_similarity_array)) == 2:
                phase_2_array.append(iteration)
                y_bert_coordinates.append(2)
                is_zero = False
            else:
                phase_3_array.append(iteration)
                y_bert_coordinates.append(3)
                is_zero = False
        else:
            if phases_similarity_array.index(max(phases_similarity_array)) == 0:
                phase_1_array.append(iteration)
                y_bert_coordinates.append(1)
            elif phases_similarity_array.index(max(phases_similarity_array)) == 1:
                phase_2_array.append(iteration)
                y_bert_coordinates.append(2)
            else:
                phase_3_array.append(iteration)
                y_bert_coordinates.append(3)

        # BERT vs Human Coding Part
        '''
        phase_0 = False
        phase_1 = False
        phase_2 = False
        phase_3 = False

        phase_index = 0

        count = 0

        if len(y_human_coordinates) == 0 or y_human_coordinates.count(0) == len(y_human_coordinates):
            while not phase_0 and phase_index < len(phase_0_words_sep):
                count = count + 1
                print("Phase 0 words", phase_0_words_sep[phase_index])
                # print("New line:", new_line)
                if phase_0_words_sep[phase_index] in new_line:
                    phase_0 = True
                phase_index = phase_index + 1

            print("Count", count)

            phase_index = 0

        while not phase_1 and phase_index < len(phase_1_words_sep):
            if phase_1_words_sep[phase_index] in new_line:
                phase_1 = True
            phase_index = phase_index + 1

        phase_index = 0

        while not phase_2 and phase_index < len(phase_2_words_sep):
            if phase_2_words_sep[phase_index] in new_line:
                phase_2 = True
            phase_index = phase_index + 1

        phase_index = 0

        while not phase_3 and phase_index < len(phase_3_words_sep):
            if phase_3_words_sep[phase_index] in new_line:
                phase_3 = True
            phase_index = phase_index + 1

        if not phase_0 and not phase_1 and not phase_2 and not phase_3:
            if iteration == 0:
                y_human_coordinates.append(0)
            else:
                y_human_coordinates.append(y_human_coordinates[len(y_human_coordinates) - 1])
        elif phase_0 and not phase_1 and not phase_2 and not phase_3:
            y_human_coordinates.append(0)
        elif not phase_0 and phase_1 and not phase_2 and not phase_3:
            y_human_coordinates.append(1)
        elif not phase_0 and not phase_1 and phase_2 and not phase_3:
            y_human_coordinates.append(2)
        elif not phase_0 and not phase_1 and not phase_2 and phase_3:
            y_human_coordinates.append(3)
        else:
            if iteration == 0:
                y_human_coordinates.append(0)
            else:
                y_human_coordinates.append(y_human_coordinates[len(y_human_coordinates) - 1])
        '''

        # BERT vs Human Coding end

        '''
        if y_human_coordinates[len(y_human_coordinates) - 1] == y_bert_coordinates[len(y_bert_coordinates) - 1]:
            matching_iteration_array.append(iteration + 1)
            y_matching_coordinates.append(y_human_coordinates[len(y_human_coordinates) - 1])
        else:
            bert_iteration_array.append(iteration + 1)
            y_bert.append(y_bert_coordinates[len(y_bert_coordinates) - 1])

            human_iteration_array.append(iteration + 1)
            y_human.append(y_human_coordinates[len(y_human_coordinates) - 1])

        '''
        print("Reached")
        # bert_iteration_array.append(iteration + 1)
        # y_bert.append(y_bert_coordinates[len(y_bert_coordinates) - 1] + 0.1)

        # human_iteration_array.append(iteration + 1)
        # y_human.append(y_human_coordinates[len(y_human_coordinates) - 1] - 0.1)

        print("Reached")
        iteration = iteration + 1
        iteration_array.append(iteration)

    # Average similarity graphing start

    print("Average Similarity Array:", average_similarity_array)

    plt.plot(iteration_array, average_similarity_array, color='black')
    plt.ylim(0, 1)

    print("phase_0_array:", phase_0_array)
    print("phase_1_array:", phase_1_array)
    print("phase_2_array:", phase_2_array)
    print("phase_3_array:", phase_3_array)

    for index in range(len(phase_0_array)):
        if index == 0:
            plt.axvline(x=phase_0_array[index] + 1, ymax=average_similarity_array[phase_0_array[index]],
                        color='darkgray',
                        label='Phase 0')
        else:
            plt.axvline(x=phase_0_array[index] + 1, ymax=average_similarity_array[phase_0_array[index]],
                        color='darkgray')
    for index in range(len(phase_1_array)):
        if index == 0:
            plt.axvline(x=phase_1_array[index] + 1, ymax=average_similarity_array[phase_1_array[index]],
                        color='royalblue',
                        label='Phase 1')
        else:
            plt.axvline(x=phase_1_array[index] + 1, ymax=average_similarity_array[phase_1_array[index]],
                        color='royalblue')
    # glitch in code
    for index in range(len(phase_2_array)):
        if index == 0:
            plt.axvline(x=phase_2_array[index] + 1, ymax=average_similarity_array[phase_2_array[index]],
                        color='olivedrab', label='Phase 2')
        else:
            plt.axvline(x=phase_2_array[index] + 1, ymax=average_similarity_array[phase_2_array[index]],
                        color='olivedrab')
    for index in range(len(phase_3_array)):
        if index == 0:
            plt.axvline(x=phase_3_array[index] + 1, ymax=average_similarity_array[phase_3_array[index]],
                        color='goldenrod', label='Phase 3')
        else:
            plt.axvline(x=phase_3_array[index] + 1, ymax=average_similarity_array[phase_3_array[index]],
                        color='goldenrod')

    plt.legend()

    plt.xlabel('Window Number')
    plt.ylabel('Similarity')
    plt.title(filename + " Similarity, window = 5")
    plt.savefig(filename + " w2v - 5in1 - words" + ".png")
    plt.clf()






