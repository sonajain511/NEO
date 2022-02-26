import scipy
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

sbert_model = SentenceTransformer('nli-bert-large')

def generate_graph_23(line, filename, phases_by_human):
    # set window size
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

    phase_0_sentence = sbert_model.encode(phase_0_words)[0]
    phase_1_sentence = sbert_model.encode(phase_1_words)[0]
    phase_2_sentence = sbert_model.encode(phase_2_words)[0]
    phase_3_sentence = sbert_model.encode(phase_3_words)[0]

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

    human_phases_array = []
    # print(human_phases_array)

    print("Length of line:", len(line))
    while iteration < len(line) - window + 1 - 2:  # don't evaluate last line --> extra - 1
        new_line = line[iteration + 1] + line[iteration + 2] + \
                   line[iteration + 3] + line[iteration + 4] + line[iteration + 5]
        if window > 2:
            if iteration == 0:
                for i in range(1, window + 1):
                    human_phases_array.append(phases_by_human[i])
                    for j in range(1, window + 1):
                        if i < j:
                            sentence_pairs_array.append((i, j))
                if is_zero:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_0_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, sbert_model.encode([new_line])[0])]
                else:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, sbert_model.encode([new_line])[0])]

                # print("Sentence Pairs Array:", sentence_pairs_array)
                # average_similarity_array.append(similarity_calculations(sentence_pairs_array, line))
            else:
                print("Human_Phases_Array:", human_phases_array)
                human_phases_array.pop(0)
                human_phases_array.append(phases_by_human[iteration + window + 1])
                print("Iteration + window + 1:", iteration + window + 1)

                # get rid of first line of matrix, since no longer relevant
                # phases_similarity_array.pop(0)
                if is_zero:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_0_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, sbert_model.encode([new_line])[0])]
                else:
                    phases_similarity_array = [
                        1 - scipy.spatial.distance.cosine(phase_1_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_2_sentence, sbert_model.encode([new_line])[0]),
                        1 - scipy.spatial.distance.cosine(phase_3_sentence, sbert_model.encode([new_line])[0])]

                # window_phases_array.pop(0)
                # window_phases_array.append(phases_similarity_matrix[len(phases_similarity_matrix) - 1].index(max(phases_similarity_matrix[len(phases_similarity_matrix) - 1])))

            contained_phases = set(human_phases_array)  # gets unique values
            contained_phases = list(contained_phases)  # converts back to list

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


        else:
            print("Iteration:", line[iteration])
            query_prev_sent = sbert_model.encode(line[iteration + 1])
            query_curr_sent = sbert_model.encode(line[iteration + 2])
            average_similarity_array.append(1 - scipy.spatial.distance.cosine(query_prev_sent, query_curr_sent))
            # modify this to incorporate phases_similarity_array if end up testing window = 2

            if is_zero:
                phases_similarity_array = [
                    1 - scipy.spatial.distance.cosine(phase_0_sentence, sbert_model.encode([new_line])[0]),
                    1 - scipy.spatial.distance.cosine(phase_1_sentence, sbert_model.encode([new_line])[0]),
                    1 - scipy.spatial.distance.cosine(phase_2_sentence, sbert_model.encode([new_line])[0]),
                    1 - scipy.spatial.distance.cosine(phase_3_sentence, sbert_model.encode([new_line])[0])]
            else:
                phases_similarity_array = [
                    1 - scipy.spatial.distance.cosine(phase_1_sentence, sbert_model.encode([new_line])[0]),
                    1 - scipy.spatial.distance.cosine(phase_2_sentence, sbert_model.encode([new_line])[0]),
                    1 - scipy.spatial.distance.cosine(phase_3_sentence, sbert_model.encode([new_line])[0])]

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

        bert_iteration_array.append(iteration + 1)
        y_bert.append(y_bert_coordinates[len(y_bert_coordinates) - 1] + 0.1)

        human_iteration_array.append(iteration + 1)
        y_human.append(y_human_coordinates[len(y_human_coordinates) - 1] - 0.1)

        iteration = iteration + 1
        iteration_array.append(iteration)

    f = plt.figure()
    f.set_figwidth(6.4)
    f.set_figheight(3.1)

    plt.scatter(bert_iteration_array, y_bert, color='orange', label='BERT')
    plt.scatter(human_iteration_array, y_human, color='purple', label='Human Rater')

    plt.yticks([0, 1, 2, 3])
    # plt.autoscale()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel('Window Number')
    plt.ylabel('Phase')
    plt.title(filename + " - BERT vs Human Coding Results")
    plt.legend()
    plt.savefig(filename + "-" + " Human vs BERT coding, reduced scale" + ".png")
    # plt.show()
    plt.clf()
