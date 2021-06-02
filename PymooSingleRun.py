import collections
import os
import pickle
import random
import sys
import time

from src.MOEA.pymoo.MOEAD.EApymooMOEAD import EApymooMOEAD
from src.MOEA.pymoo.NSGA2.EApymooNSGAII import EApymooNSGAII
from src.MOEA.pymoo.NSGA3.EApymooNSGAIII import EApymooNSGAIII

arguments = sys.argv[1:]

if len(arguments) < 3:
    print("Please, provide the arguments ('*' = required, '-' = optional):\n"
          "\t*1) program name (see 'data/bug/dataset' for available programs in '.pkl' format);\n"
          "\t*2) independent run ID;\n"
          "\t*3) algorithm ('NSGAII', 'NSGAIII', or 'MOEAD');\n"
          "\t-4) objectives ('UnRec-Size', 'UnObs-Size', 'UnRec-UnObs', or 'UnRec-Size-UnObs'). Bear in mind that 2 "
          "objectives are only implemented with NSGAII. Default: 'UnRec-Size-UnObs';\n"
          "\t-5) random seed")
    exit(1)

start = time.time()

program = arguments[0]
run_id = arguments[1]
algorithm = arguments[4]

if len(arguments) < 4 or algorithm != "NSGAII":
    objectives = "UnRec-Size-UnObs"
else:
    objectives = arguments[3]

seed_0 = None
if len(arguments) >= 5:
    seed_0 = arguments[4]
else:
    seed_0 = os.urandom(128)
random.seed(seed_0)

F = None
try:
    F = open('data/bug/dataset/BugStepsMappedPp_' + program + '_plus.pkl', 'rb')
except FileNotFoundError:
    print("Could not find the dataset file 'data/bug/dataset/BugStepsMappedPp_" + program + "_plus.pkl'.")
    exit(1)

if algorithm not in ["NSGAII", "NSGAIII", "MOEAD"]:
    print("Invalid algorithm. Please provide one of the following: 'NSGAII', 'NSGAIII', or 'MOEAD'")
    exit(1)

if objectives not in ["UnRec-Size", "UnObs-Size", "UnRec-UnObs", "UnRec-Size-UnObs"]:
    print("Invalid combination of objectives. Please, provide one of the following: UnRec-Size, UnObs-Size, "
          "UnRec-UnObs, or UnRec-Size-UnObs")
    exit(1)

results_dir = 'results/' + program + '/' + algorithm + '/' + objectives + "/"
os.makedirs(os.path.dirname(results_dir), exist_ok=True)

step_map = pickle.load(F)
F.close()

step_map_train = collections.OrderedDict()
key_s = list(step_map.keys())
for i in range(round((len(step_map) * 80) / 100)):
    step_map_train[key_s[i]] = step_map[key_s[i]]

EA = None
if algorithm == "NSGAII":
    EA = EApymooNSGAII(run_id, program, step_map_train, objectives)
if algorithm == "NSGAIII":
    EA = EApymooNSGAIII(run_id, program, step_map_train)
if algorithm == "MOEAD":
    EA = EApymooMOEAD(run_id, program, step_map_train)

EA.run()
del EA

with open(results_dir + "run_" + run_id + "_time.txt", "w") as timefile:
    end = time.time()
    print(f"{end - start}", file=timefile)
