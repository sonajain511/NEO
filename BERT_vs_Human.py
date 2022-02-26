import scipy
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

sbert_model = SentenceTransformer('nli-bert-large')


def generate_graph(line, filename):
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
                #average_similarity_array.append(similarity_calculations(sentence_pairs_array, line))
            else:
                #print("Human_Phases_Array:", human_phases_array)
                #human_phases_array.pop(0)
                #human_phases_array.append(phases_by_human[iteration + window + 1])
                print("Iteration + window + 1:", iteration + window + 1)

                ''''
                for i in range(0, window - 1):
                    sentence_pairs_array.pop(0)
                # originally iteration
                sentence_pairs_array.append(retrieve_next_sentence_array(iteration + 1, window - 2, sentence_pairs_array))
                # likes to add [...]
                sentence_pairs_array.pop()
                average_similarity_array.append(similarity_calculations(sentence_pairs_array, line))
                '''

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

        # BERT vs Human Coding Part

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

        bert_iteration_array.append(iteration + 1)
        y_bert.append(y_bert_coordinates[len(y_bert_coordinates) - 1] + 0.1)

        human_iteration_array.append(iteration + 1)
        y_human.append(y_human_coordinates[len(y_human_coordinates) - 1] - 0.1)

        iteration = iteration + 1
        iteration_array.append(iteration)

    # Average similarity graphing start

    ''''
    print("Average Similarity Array:", average_similarity_array)

    plt.plot(iteration_array, average_similarity_array, color = 'black')
    plt.ylim(0, 1)

    print("phase_0_array:", phase_0_array)
    print("phase_1_array:", phase_1_array)
    print("phase_2_array:", phase_2_array)
    print("phase_3_array:", phase_3_array)



    for index in range(len(phase_0_array)):
        if index == 0:
            plt.axvline(x=phase_0_array[index] + 1, ymax=average_similarity_array[phase_0_array[index]], color='darkgray',
                        label='Phase 0')
        else:
            plt.axvline(x=phase_0_array[index] + 1, ymax=average_similarity_array[phase_0_array[index]], color='darkgray')
    for index in range(len(phase_1_array)):
        if index == 0:
            plt.axvline(x=phase_1_array[index] + 1, ymax=average_similarity_array[phase_1_array[index]], color='royalblue',
                        label='Phase 1')
        else:
            plt.axvline(x=phase_1_array[index] + 1, ymax=average_similarity_array[phase_1_array[index]], color='royalblue')
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
    plt.savefig(filename + " NLI - 5in1 - words" + ".png")
    plt.clf()
    '''

    '''
    phase_0 = False
    phase_1 = False
    phase_2 = False
    phase_3 = False

    phase_index = 0


    if y_human_coordinates == [] or y_human_coordinates.count(0) == len(y_human_coordinates):
        while not phase_0 and phase_index < len(phase_0_words):
            if phase_0_words[phase_index] in new_line:
                phase_0 = True
            phase_index = phase_index + 1

    phase_index = 0

    while not phase_1 and phase_index < len(phase_1_words):
        if phase_1_words[phase_index] in new_line:
            phase_1 = True
        phase_index = phase_index + 1

    phase_index = 0

    while not phase_2 and phase_index < len(phase_2_words):
        if phase_2_words[phase_index] in new_line:
            phase_2 = True
        phase_index = phase_index + 1

    phase_index = 0

    while not phase_3 and phase_index < len(phase_3_words):
        if phase_3_words[phase_index] in new_line:
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

    # print("BERT coordinates:", bert_coordinates)

    # Human vs BERT graphing start

    # y_human_coordinates_mod = y_human_coordinates

    # plt.rcParams["figure.figsize"] = [6.4, 4.2]
    f = plt.figure()
    f.set_figwidth(6.4)
    f.set_figheight(3.1)

    plt.scatter(bert_iteration_array, y_bert, color='orange', label='BERT')
    plt.scatter(human_iteration_array, y_human, color='purple', label='Human Rater')
    # plt.scatter(matching_iteration_array, y_matching_coordinates, color = 'black', label = 'Overlap')

    '''
    human_count = 0
    bert_count = 0
    overlap_count = 0


    for index in range(len(bert_iteration_array) - 1):
        if (y_bert[index] == y_bert[index + 1]) and bert_iteration_array[index] == bert_iteration_array[index + 1] - 1:
            bert_count = bert_count + 1
            if bert_count == 1:
                plt.plot(iteration_array[index:index+2], y_bert[index:index+2], color = 'orange', label = 'BERT')
            else:
                plt.plot(iteration_array[index:index + 2], y_bert[index:index + 2], color='orange')

    for index in range(len(human_iteration_array) - 1):
        if (y_human[index] == y_human[index + 1]) and human_iteration_array[index] == human_iteration_array[index + 1] - 1:
            human_count = human_count + 1
            if human_count == 1:
                plt.plot(human_iteration_array[index:index + 2], y_human[index:index + 2], color='purple',
                         label='Human Rater')
            else:
                plt.plot(human_iteration_array[index:index + 2], y_human[index:index + 2], color='purple')

    for index in range(len(y_matching_coordinates) - 1):
        if (y_matching_coordinates[index] == y_matching_coordinates[index + 1]) and matching_iteration_array[index] == matching_iteration_array[index + 1] - 1:
            overlap_count = overlap_count + 1
            if overlap_count == 1:
                plt.plot(matching_iteration_array[index:index + 2], y_matching_coordinates[index:index + 2], color='black',
                         label='Overlap')
            else:
                plt.plot(matching_iteration_array[index:index + 2], y_matching_coordinates[index:index + 2], color='black')

    '''
    '''

    figure, axs = plt.subplots(2)
    plt.rc('axes', labelsize = 30)
    figure.suptitle(filename + " - BERT vs Human Coding Results")
    figure.text(0.5, 0.04, 'Window Number', ha='center', va='center')
    figure.text(0.06, 0.5, 'Phase', ha='center', va='center', rotation='vertical')

    # ax1 = figure.add_subplot(221)
    # ax2 = figure.add_subplot(222, sharex = ax1, sharey = ax1)

    # iteration array, instead of x-bert coordinates

    axs[0].scatter(iteration_array, y_bert_coordinates, color='orange', s=3, label = 'BERT Phase')
    axs[0].set_ylim((-0.25,3.25))
    axs[0].legend(loc='lower right')

    # axs[0].xlabel("Window Number")
    # axs[0].label("BERT Phase")

    for axis in [axs[0].xaxis, axs[0].yaxis]:
      axis.set_major_locator(ticker.MaxNLocator(integer=True))

    for index in range(len(iteration_array) - 1):
        if (y_bert_coordinates[index] == y_bert_coordinates[index + 1]):
            axs[0].plot(iteration_array[index:index+2], y_bert_coordinates[index:index+2], color = 'orange')

    axs[1].scatter(iteration_array, y_human_coordinates_mod, color='purple', s=3, label = 'Human Rater Phase')
    axs[1].set_ylim((-0.25, 3.25))
    axs[1].legend(loc = 'lower right')

    # axs[1].xlabel("Window Number")
    # axs[1].label("Human Rater Phase")

    for axis in [axs[1].xaxis, axs[1].yaxis]:
      axis.set_major_locator(ticker.MaxNLocator(integer=True))

    for index in range(len(iteration_array) - 1):
        if (y_human_coordinates_mod[index] == y_human_coordinates_mod[index + 1]):
            axs[1].plot(iteration_array[index:index + 2], y_human_coordinates_mod[index:index + 2], color='purple')

    '''

    # plt.plot(x_bert_coordinates, y_bert_coordinates)

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