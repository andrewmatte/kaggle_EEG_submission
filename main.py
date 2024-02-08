from multiprocessing import Process
import numpy as np
import os
import pandas as pd
from random import shuffle
import xgboost as xgb


# CONSTANT
classifications = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]


# DATA ACCESS FUNCTIONS
def get_metadata_from_sources(source_file, min_votes=0):
    train = open(source_file, "r").read().split("\n")
    train.pop(0)
    train.pop()
    EEG_ids_to_offset = {}
    num_votes_list = []
    for line in train:
        ids = line.split(",")
        if len(ids) < 5:
            try:
                EEG_ids_to_offset[ids[1]] = [{"offset": 0, "expert_consensus": None}]
            except:
                EEG_ids_to_offset[ids[1]].append(
                    {"offset": 0, "expert_consensus": None}
                )
        else:
            num_votes = sum([int(num) for num in ids[-6:]])
            max_vote = max([int(num) for num in ids[-6:]])
            if max_vote == num_votes and num_votes >= min_votes:
                num_votes_list.append(num_votes)
                if ids[0] not in EEG_ids_to_offset:
                    EEG_ids_to_offset[ids[0]] = [
                        {"offset": float(ids[2]), "expert_consensus": ids[-7]}
                    ]
                else:
                    EEG_ids_to_offset[ids[0]].append(
                        {"offset": float(ids[2]), "expert_consensus": ids[-7]}
                    )
    return EEG_ids_to_offset


def make_folder(OUTPUT_FOLDER):
    try:
        os.mkdir(OUTPUT_FOLDER)
    except:
        Exception("folder exists")


def create_segment(all_waves, chunk_size, offset):
    del all_waves["EKG"]
    return all_waves[int(offset) * 200 : int(offset) * 200 + chunk_size]


def get_segments_with_offset_from_folder(SOURCE_FILE, SOURCE_FOLDER):
    EEG_ids_to_offset = get_metadata_from_sources(SOURCE_FILE, 10)
    filenames = set(os.listdir(SOURCE_FOLDER))
    eeg_ids = []
    for eeg_id in EEG_ids_to_offset:
        for offset in EEG_ids_to_offset[eeg_id]:
            if eeg_id + "_" + str(offset) + ".txt" not in filenames:
                eeg_ids.append([eeg_id, offset])
    return eeg_ids


def transform_data_from_source_to_folder(eeg_ids, OUTPUT_FOLDER, SOURCE_FOLDER):
    count = 0
    for eeg_id in eeg_ids:
        eeg_id.append(OUTPUT_FOLDER)
        eeg_id.append(SOURCE_FOLDER)
        analyze_one_eeg_or_segment(eeg_id)
        count += 1
        if count % 10 == 0:
            print(count, "of", len(eeg_ids))


def get_mapping_from_output_folder(OUTPUT_FOLDER):
    mapping = {}
    filenames = os.listdir(OUTPUT_FOLDER)
    try:
        substring = OUTPUT_FOLDER.lower().index("test")
    except:
        substring = -1
    if substring > -1:
        for filename in filenames:
            file = open(OUTPUT_FOLDER + "/" + filename, "r").read().split("\n")
            file.pop()
            key = filename[:-4] + ",1"
            try:
                mapping[key].append(file[1])
            except:
                mapping[key] = [file[1]]
    else:
        for filename in filenames:
            file = open(OUTPUT_FOLDER + "/" + filename, "r").read().split("\n")
            file.pop()
            key = filename[:-4] + "," + file[0]
            try:
                mapping[key].append(file[1])
            except:
                mapping[key] = [file[1]]

    return mapping


def write_to_csv_from_folder_mapping(mapping, FINAL_OUTPUT_CSV):
    headers = ["ID", "event"]
    for i in range(20):
        for j in range(3):
            headers.append("amplitudes_" + str(i) + "-" + str(j))
    for i in range(3):
        headers.append("total_amplitude_" + str(i))
    for i in range(210):
        headers.append("overlap_ratios_" + str(i))
    for i in range(190):
        headers.append("correlations_" + str(i))
    for i in range(19):
        headers.append("crossover_count_" + str(i))
    open(FINAL_OUTPUT_CSV, "w").write(",".join(headers) + "\n")
    count = 0
    for key in mapping:
        count += 1
        for item in range(len(mapping[key]) - 1, -1, -1):
            try:
                mapping[key][item] = eval(mapping[key][item])
                string_lists = list(mapping[key][item][:5])
                for index in range(len(string_lists)):
                    if type(string_lists[index][0]) == list:
                        flat_list = []
                        for row in string_lists[index]:
                            for number in row:
                                flat_list.append(number)
                        string_lists[index] = flat_list
                    finalized = ""
                    for string_list in string_lists:
                        for number in string_list:
                            finalized += str(number) + ","
                open(FINAL_OUTPUT_CSV, "a").write(key + "," + finalized[:-1] + "\n")
            except Exception as e:
                mapping[key].pop(item)
                print("oopsie", e)


# FEATURE ENGINEERING FUNCTIONS
def get_amplitudes_max_min(all_waves):
    # order: min, max, amplitude
    output = []
    for axe in all_waves.axes[1]:
        output.append([all_waves[axe].min(), all_waves[axe].max()])
        output[-1].append(output[-1][1] - output[-1][0])
    return output


def get_total_amplitude_min_max(all_amplitudes):
    minimum = min([x[0] for x in all_amplitudes])
    maximum = max([x[1] for x in all_amplitudes])
    max_variance = max([x[1] - x[0] for x in all_amplitudes])
    return [minimum, maximum, max_variance]


def get_overlap_ratios_from_amplitudes(all_amplitudes):
    ratios = []
    for amp in range(len(all_amplitudes) - 1):
        for amp2 in range(amp + 1, len(all_amplitudes)):
            max_of_mins = max(all_amplitudes[amp][0], all_amplitudes[amp2][0])
            min_of_maxes = min(all_amplitudes[amp][1], all_amplitudes[amp2][1])
            if max_of_mins < min_of_maxes:
                ratios.append(
                    (min_of_maxes - max_of_mins)
                    / (
                        max(all_amplitudes[amp][1], all_amplitudes[amp2][1])
                        - min(all_amplitudes[amp][0], all_amplitudes[amp2][0])
                    )
                )
            else:
                ratios.append(0)
    return ratios


def get_all_correlations(all_waves):
    # ordered only because unordered is not excellent
    output = []
    all_axes = all_waves.axes[1]
    for axe_index in range(len(all_axes) - 1):
        for axe2_index in range(axe_index + 1, len(all_axes)):
            correlation = all_waves[all_axes[axe_index]].corr(
                all_waves[all_axes[axe2_index]]
            )
            output.append(correlation)
    return output


def get_crossover_counts(all_waves):
    output = []
    length = len(all_waves)
    all_axes = all_waves.axes[1]
    all_waves.reset_index()
    all_waves.index = pd.RangeIndex(len(all_waves.index))
    for axe_index in range(len(all_axes) - 1):
        count = 0
        for axe2_index in range(axe_index + 1, len(all_axes)):
            for index in range(length - 1):
                if (
                    all_waves[all_axes[axe_index]][index]
                    - all_waves[all_axes[axe2_index]][index]
                ) * (
                    all_waves[all_axes[axe_index]][index + 1]
                    - all_waves[all_axes[axe2_index]][index + 1]
                ) < 0:
                    count += 1
        output.append(count)
    return output


def analyze_one_eeg_or_segment(eeg_id_data):
    eeg = pd.read_parquet("./" + eeg_id_data[3] + "/" + eeg_id_data[0] + ".parquet")
    eeg = create_segment(eeg, 400, eeg_id_data[1]["offset"])
    amplitudes = get_amplitudes_max_min(eeg)
    total_amplitude = get_total_amplitude_min_max(amplitudes)
    overlap_ratios = get_overlap_ratios_from_amplitudes(amplitudes + [total_amplitude])
    correlations = get_all_correlations(eeg)
    crossover_count = get_crossover_counts(eeg)
    # very few features are needed to be able to distinguish between harmful brain activity
    # the difference between Lateralized and other brain patterns is found in the data, without the need to look at EKG data
    y = (
        amplitudes,
        total_amplitude,
        overlap_ratios,
        correlations,
        crossover_count,
    )
    try:
        open(
            eeg_id_data[2]
            + "/"
            + eeg_id_data[0]
            + "_"
            + str(int(eeg_id_data[1]["offset"]))
            + ".txt",
            "a",
        ).write(eeg_id_data[0] + "," + str(eeg_id_data[1]["expert_consensus"]) + "\n")
    except Exception as e:
        print("problem writing", e)
    open(
        eeg_id_data[2]
        + "/"
        + eeg_id_data[0]
        + "_"
        + str(int(eeg_id_data[1]["offset"]))
        + ".txt",
        "a",
    ).write(str(y) + "\n")


# PREDICTION FUNCTIONS
def total_error(p, q, ARBITRARY_METRIC=5000):
    int_predictions = []
    for item_index in range(len(p)):
        maximum = 0.0
        maximum_index = 0
        for index in range(6):
            if maximum < p[item_index][index]:
                maximum_index = index
                maximum = p[item_index][index]
            int_predictions.append([0.0] * 6)
            int_predictions[item_index][maximum_index] = 1.0
    q = q.get_label()
    chunk_size = 6
    real_answers = []
    index = 0
    while index < len(q):
        real_answers.append(q[index : index + chunk_size])
        index += chunk_size
    distance = 0
    for rownum in range(len(p)):
        distance += sum(
            [
                int_predictions[rownum][i] * real_answers[rownum][i]
                for i in range(len(int_predictions[rownum]))
            ]
        )
    return "total_error", float(ARBITRARY_METRIC - distance)


def get_data_from_csv(FINAL_OUTPUT_CSV, shuffle=False):
    try:
        substring_index = FINAL_OUTPUT_CSV.lower().index("test")
    except:
        substring_index = -1
    if substring_index != -1:
        all_data = open(FINAL_OUTPUT_CSV, "r").read().split("\n")
        all_data.pop(0)
        all_data.pop()
        all_data = [eval(a) for a in all_data]
        labels = [a[0] for a in all_data]
        X_data = [a[2:] for a in all_data]
    else:
        all_data = [
            line.split(",") for line in open(FINAL_OUTPUT_CSV, "r").read().split("\n")
        ]
        _ = all_data.pop(0)  # headings
        all_data.pop()
        if shuffle:
            shuffle(all_data)
        X_data = [d[3:] for d in all_data]
        _ = [d[1] for d in all_data]  # IDs
        Label_data = [d[2] for d in all_data]

    X_data = np.asarray([np.asarray([float(num) for num in x]) for x in X_data])
    if substring_index != -1:
        return X_data, labels
    try:
        Y_data = [[0.0 for _ in range(len(classifications))] for y in Label_data]
        for index in range(len(Y_data)):
            try:
                spot = classifications.index(Label_data[index])
            except:
                print(Label_data[index])
            Y_data[index][spot] = 1.0
            Y_data[index][spot] = np.asarray(Y_data[index][spot])

        Y_data = np.asarray(Y_data)
    except:
        Y_data = None

    return X_data, Y_data


def train_test_split(X_data, Y_data, TEST_PERCENT=2):
    TEST_AMOUNT = int((TEST_PERCENT / 100) * X_data.shape[0])
    X_train = X_data[TEST_AMOUNT:]
    X_test = X_data[:TEST_AMOUNT]
    if Y_data is not None:
        y_train = Y_data[TEST_AMOUNT:]
        y_test = Y_data[:TEST_AMOUNT]
        Xy_test = xgb.DMatrix(X_test, y_test)
        Xy = xgb.DMatrix(X_train, y_train)
        return Xy, Xy_test
    else:
        Xy = xgb.DMatrix(X_train, y_train)
        return Xy, None


def secondary_trainer(X, Y, rounds=25):
    clf = xgb.XGBClassifier(eval_metric=total_error, verbosity=1)
    clf.fit(X, Y)
    return clf


def train_on_data(Xy, Xy_test, rounds=25):
    booster = xgb.train(
        {"tree_method": "hist"},
        dtrain=Xy,
        num_boost_round=rounds,
        custom_metric=total_error,
        evals=[(Xy, "train")],
    )
    return booster


def predict_on_data(model, data):
    predictions = model.predict(data)
    return predictions


def print_total_error_categories(p, q):
    int_predictions = []
    for item_index in range(len(p)):
        maximum = 0.0
        maximum_index = 0
        for index in range(6):
            if maximum < p[item_index][index]:
                maximum_index = index
                maximum = p[item_index][index]
            int_predictions.append([0.0] * 6)
            int_predictions[item_index][maximum_index] = 1.0
    q = q.get_label()
    chunk_size = 6
    real_answers = []
    index = 0
    while index < len(q):
        real_answers.append(q[index : index + chunk_size])
        index += chunk_size
    category_error = {}
    for rownum in range(len(p)):
        score = 0
        for i in range(len(int_predictions[rownum])):
            score += int_predictions[rownum][i] * real_answers[rownum][i]
        if score > 0:
            try:
                category_error[np.where(real_answers[rownum] == 1.0)[0][0]][
                    "correct"
                ] += 1
            except:
                category_error[np.where(real_answers[rownum] == 1.0)[0][0]] = {
                    "correct": 1,
                    "error": 0,
                }
        else:
            try:
                category_error[np.where(real_answers[rownum] == 1.0)[0][0]][
                    "error"
                ] += 1
            except:
                category_error[np.where(real_answers[rownum] == 1.0)[0][0]] = {
                    "correct": 0,
                    "error": 1,
                }
    for cat in category_error:
        category_error[cat]["ratio"] = category_error[cat]["correct"] / (
            category_error[cat]["correct"] + category_error[cat]["error"]
        )
        print(classifications[int(cat)], category_error[cat])


# AGGREGATE FUNCTIONS
def train_and_predict(csv_filename, booster=None, train=True):
    if train:
        X, Y = get_data_from_csv(csv_filename)
        train_data, test_data = train_test_split(X, Y)
        booster = secondary_trainer(X, Y)
        # booster = train_on_data(train_data, test_data, 30)
        # predictions = predict_on_data(booster, train_data)
        # print_total_error_categories(predictions, train_data)
        return booster
    else:
        X, _ = get_data_from_csv(csv_filename)
        # X = xgb.DMatrix(X)
        predictions = predict_on_data(booster, X)
        return _, predictions


def data_pipeline_before_model(
    csv_output_filename,
    SOURCE_FILE,
    SOURCE_FOLDER,
    intermediary_folder_name,
):
    make_folder(intermediary_folder_name)
    segment_list = get_segments_with_offset_from_folder(SOURCE_FILE, SOURCE_FOLDER)
    transform_data_from_source_to_folder(
        segment_list, intermediary_folder_name, SOURCE_FOLDER
    )
    mapping = get_mapping_from_output_folder(intermediary_folder_name)
    write_to_csv_from_folder_mapping(mapping, csv_output_filename)


def output_predictions_to_file(labels, predictions):
    headers = "eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote"
    output = []
    for index in range(len(labels)):
        output.append(
            ",".join(
                [str(labels[index])] + [str(float(num)) for num in predictions[index]]
            )
        )
    filedata = headers + "\n" + "\n".join(output)
    open(WORKING_PATH + "submission.csv", "w").write(filedata)


# choose your base_path depending on your environment
BASE_PATH = "../input/hms-harmful-brain-activity-classification/"
BASE_PATH = ""
WORKING_PATH = ""


def main():
    OUTPUT_FILENAME = WORKING_PATH + "TRAINING_DATA.csv"
    SOURCE_FILE = BASE_PATH + "train.csv"
    SOURCE_FOLDER = BASE_PATH + "train_eegs"
    # data_pipeline_before_model(
    #     OUTPUT_FILENAME, SOURCE_FILE, SOURCE_FOLDER, WORKING_PATH + "TRAIN_MIDWAY"
    # )
    model = train_and_predict(OUTPUT_FILENAME)
    OUTPUT_FILENAME = WORKING_PATH + "TESTING_DATA.csv"
    SOURCE_FILE = BASE_PATH + "test.csv"
    SOURCE_FOLDER = BASE_PATH + "test_eegs"
    data_pipeline_before_model(
        OUTPUT_FILENAME, SOURCE_FILE, SOURCE_FOLDER, WORKING_PATH + "TEST_MIDWAY"
    )
    labels, predictions = train_and_predict(OUTPUT_FILENAME, model, train=False)
    output_predictions_to_file(labels, predictions)


if __name__ == "__main__":
    main()
