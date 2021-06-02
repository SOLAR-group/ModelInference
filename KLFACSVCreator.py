import pickle
prod = ["kate", "Vibe", "krita", "LibreOffice", "Firefox OS", "Firefox for Android", "SeaMonkey", "Thunderbird",
        "Calendar", "BIRT"]

for product in prod:
    infile = open("/home/francali/Documents/MOGA_NLP_testing/testModels-bugReports/data/bug/dataset/BugStepsMappedPp_"
                  + product + "_plus.pkl", "rb")
    out_train_file = open("/home/francali/Documents/MOGA_NLP_testing/klfa-src/examples/glassfishForumUserIssue/bin/datasets/train/"
                          + product + "_plus.csv", "w")
    out_test_file = open("/home/francali/Documents/MOGA_NLP_testing/klfa-src/examples/glassfishForumUserIssue/bin/datasets/test/"
                         + product + "_plus.csv", "w")
    steps = pickle.load(infile)

    keys = list(steps.keys())
    for i in range(round((len(keys) * 80) / 100)):
        trace = steps[keys[i]]
        for step in trace:
            out_train_file.write("0,"+step+"\n")
        if i != (round((len(keys) * 80) / 100) - 1):
            out_train_file.write("|\n")

    for i in range(round((len(keys) * 80) / 100), len(keys)):
        trace = steps[keys[i]]
        for step in trace:
            out_test_file.write("0,"+step+"\n")
        if i != (len(keys)-1):
            out_test_file.write("|\n")
