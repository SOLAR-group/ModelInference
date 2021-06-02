import datetime
import time
import pickle
import collections
from statistics import variance, stdev


class DatasetsInfo:

    def __init__(self, product, steps):
        self.product = product
        self.bugTraces = steps

        self.step_map_train = collections.OrderedDict()
        key_s = list(self.bugTraces.keys())
        for i in range(round((len(self.bugTraces) * 80) / 100)):
            self.step_map_train[key_s[i]] = self.bugTraces[key_s[i]]

        self.step_map_test = collections.OrderedDict()
        key_s = list(self.bugTraces.keys())
        for i in range(round((len(self.bugTraces) * 80) / 100), len(self.bugTraces)):
            self.step_map_test[key_s[i]] = self.bugTraces[key_s[i]]

    def computeInfo(self):
        info = list()

        keys = self.step_map_train.keys()
        num_bugs = len(keys)
        bug_length_sum = 0
        bug_lengths = list()
        for key in keys:
            bugTrace = self.step_map_train[key]
            bug_lengths.append(len(bugTrace))
            bug_length_sum += len(bugTrace)
        mean_bug_length = bug_length_sum/num_bugs
        info.append([round(stdev(bug_lengths), 2), round(variance(bug_lengths), 2), round(mean_bug_length, 2), num_bugs])

        keys = self.step_map_test.keys()
        num_bugs = len(keys)
        bug_length_sum = 0
        bug_lengths = list()
        for key in keys:
            bugTrace = self.step_map_test[key]
            bug_lengths.append(len(bugTrace))
            bug_length_sum += len(bugTrace)
        mean_bug_length = bug_length_sum / num_bugs
        info.append([round(stdev(bug_lengths), 2), round(variance(bug_lengths), 2), round(mean_bug_length, 2), num_bugs])

        return info


if __name__ == '__main__':
    # Datasets are ordered by size


    software =     ["kate", "Vibe", "krita", "LibreOffice", "Firefox OS",
    "Firefox for Android", "SeaMonkey", "Thunderbird", "Calendar", "BIRT"]
    startf = time.perf_counter()
    for product in software:
        print(product + "\n")
        steps_file = open('../../../data/bug/dataset/BugStepsMappedPp_' + product + '_plus.pkl', 'rb')
        step_map = pickle.load(steps_file)
        steps_file.close()
        datasetInfo = DatasetsInfo(product, step_map)
        info = datasetInfo.computeInfo()
        print("Train")
        print("Standard Deviation: " + str(info[0][0]) +
              "\nVariance: " + str(info[0][1]) +
              "\nMean: " + str(info[0][2]) +
              "\nN. of bugs: " + str(info[0][3]))
        print("\nTest")
        print("Standard Deviation: " + str(info[1][0]) +
              "\nVariance: " + str(info[1][1]) +
              "\nMean: " + str(info[1][2]) +
              "\nN. of bugs: " + str(info[1][3]))
    print("Done. Time: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))
