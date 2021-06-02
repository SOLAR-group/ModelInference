from src.MOEA.deap.EAdeapGA import *
from src.MOEA.deap.EAdeapNSGAII import *
from src.MOEA.deap.EAdeapNSGAIII import *
import time
import gc
import pickle
import random
import multiprocessing
import datetime
import os
import collections

from src.MOEA.pymoo.MOEAD.EApymooMOEAD import EApymooMOEAD
from src.MOEA.pymoo.NSGA2.EApymooNSGAII import EApymooNSGAII
from src.MOEA.pymoo.NSGA3.EApymooNSGAIII import EApymooNSGAIII


def exec_run(n_run, product, step_map_train):
    seed_0 = os.urandom(128)

    # random.seed(seed_0)
    # startf = time.process_time()
    # print("Product " + program + " Start NSGAIII run_id: " + str(n_run))
    # EA = EApymooNSGAIII(n_run, program, step_map_train)
    # EA.run_id()
    # del EA
    # print("Product " + program + " End NSGAIII run_id: " + str(n_run) + " Time tot (gen + minH pareto): " + str(
    #     datetime.timedelta(seconds=round(time.process_time() - startf))))

    random.seed(seed_0)
    startf = time.process_time()
    print("Product " + product + " Start MOEA/D run_id: " + str(n_run))
    EA = EApymooMOEAD(n_run, product, step_map_train)
    EA.run()
    del EA
    print("Product " + product + " End MOEA/D run_id: " + str(n_run) + " Time tot (gen + minH pareto): " + str(
        datetime.timedelta(seconds=round(time.process_time() - startf))))

    # random.seed(seed_0)
    # startf = time.process_time()
    # print("Product " + program + " Start NSGAII run_id: "+str(n_run))
    # EA = EApymooNSGAII(n_run, program, step_map_train, "UnRec-Size-UnObs")
    # EA.run_id()
    # del EA
    # print("Product " + program + " End NSGAII run_id: "+str(n_run)+" Time tot (gen + minH pareto): " + str(
    #     datetime.timedelta(seconds=round(time.process_time() - startf))))

    # random.seed(seed_0)
    # startf = time.process_time()
    # print("Product " + program + " Start GA run_id: " + str(n_run))
    # EA = EAdeapGA(n_run, program, step_map_train)
    # EA.compute()
    # del EA
    # print("Product " + program + " End GA run_id: " + str(n_run) + " Time tot (gen + minH pareto): " + str(
    #     datetime.timedelta(seconds=round(time.process_time() - startf))))


if __name__ == '__main__':
    # Datasets are ordered by size
    prod = ["kate", "Vibe", "krita", "LibreOffice", "Firefox OS", "Firefox for Android", "SeaMonkey", "Thunderbird", "Calendar", "BIRT"]
    startf = time.perf_counter()
    step_map_all = list()
    div_p_all = list()
    prod_names_all = list()

    for product in prod:
        # results_dir = 'data/'+program+'/MHGA/'
        # os.makedirs(os.path.dirname(results_dir), exist_ok=True)
        # results_dir = 'data/' + program + '/NSGAII/'
        # os.makedirs(os.path.dirname(results_dir), exist_ok=True)
        # results_dir = 'data/' + program + '/NSGAIII/'
        # os.makedirs(os.path.dirname(results_dir), exist_ok=True)
        filename = 'data/' + product + '/MOEAD/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        F = open('data/bug/dataset/BugStepsMappedPp_'+product+'_plus.pkl', 'rb')
        step_map = pickle.load(F)
        F.close()

        step_map_train = collections.OrderedDict()
        key_s = list(step_map.keys())
        for i in range(round((len(step_map) * 80) / 100)):
            step_map_train[key_s[i]] = step_map[key_s[i]]

        #print("Num bug trace training: "+str(round((len(step_map) * 80) / 100)))

        num_itr = 30  # 30

        for n_itr in range(1, num_itr + 1):
            div_p_all.append(n_itr)
            prod_names_all.append(product)
            step_map_all.append(step_map_train)

    num_pr = 1  # 8
    jobs = list(zip(div_p_all, prod_names_all, step_map_all))
    pool = multiprocessing.Pool(num_pr)
    pool.starmap(exec_run, jobs, chunksize=1)
    pool.close()
    pool.terminate()

    print("Done. Time: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))
