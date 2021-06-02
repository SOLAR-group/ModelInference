from src.MOEA.deap.FSMop import *
from deap import creator, base, tools
import time
import pickle
import random
import datetime
import collections


class TestR:
    def __init__(self, prod, step):
        self.product = prod
        self.step_map = step
        self.fsm = None

    def createIndivD(self):
        id = self.counter
        self.counter += 1
        if id >= len(self.fsm.dfaList):
            rn = random.randint(0, len(self.fsm.dfaList) - 1)
            return self.fsm.dfaList[rn].dup()
        else:
            return self.fsm.dfaList[id].dup()

    def evaluateI(self, individual):
        return self.fsm.UnRec(individual), len(individual[0].States), self.fsm.UnObs(individual[0])

    def rtr_sigma(self, dfa):
        sgm = set()
        for k in dfa.delta:
            for key in dfa.delta[k]:
                sgm.add(key)
        dfa.Sigma = sgm

    def compute_res(self):
        #3 is the time, 4 is the number of solution, 5 is avg bug for pareto, 6 tot of bug
        alg_tris = {"GA":[[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], 0, 0, 0, 0], "NSGAII": [[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], 0, 0, 0, 0]}
        ga_all = {"GA": [[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], 0, 0, 0, 0]}

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("cdfa", self.createIndivD)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.cdfa, 1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluateI)
        toolbox.register("map", map)

        step_map_train = collections.OrderedDict()
        key_s = list(self.step_map.keys())
        for i in range(round((len(self.step_map) * 80) / 100)):
            step_map_train[key_s[i]] = self.step_map[key_s[i]]
        print("Num bug trace training: " + str(round((len(self.step_map) * 80) / 100)))
        fsm_r = FSMop(step_map_train)

        step_map_test = collections.OrderedDict()
        key_s = list(self.step_map.keys())
        for i in range(round((len(self.step_map) * 80) / 100), len(self.step_map)):
            step_map_test[key_s[i]] = self.step_map[key_s[i]]
        fsm_t = FSMop(step_map_test)
        self.fsm = fsm_t
        print("Num bug trace test: " + str(len(self.step_map) - round((len(self.step_map) * 80) / 100)))
        start = time.perf_counter()
        sgm = (fsm_t.sigma).union(fsm_r.sigma)
        #fold = '/media/galahad/Documents/Downloads/EXP/'+program
        F = open('data/autom/Pff.pkl', 'rb')

        pareto = pickle.load(F)
        F.close()
        print(str(pareto.countTransitions()))

        j = 0
        pareto.Sigma = sgm
        #pareto = pareto[0]
        print(len(pareto.States))
        for st in fsm_t.step_listSP:
            if pareto.evalWordP(st, pareto.Initial):
                j += 1
        print(str(j))
        print(str(self.fsm.UnObs2(pareto)))
        #dd = fsm_t.traces(pareto)
        #print(str(dd))

if __name__ == '__main__':

    software = ["Firefox for Android"]
    startf = time.perf_counter()
    step = list()
    for product in software:
        print(product + "\n")

        filename = 'data/' + product + '/GA/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = 'data/' + product + '/NSGAII/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        F = open('data/bug/dataset/BugStepsMappedPp_' + product + '_plus.pkl', 'rb')
        step_map = pickle.load(F)
        F.close()
        ts = TestR(product, step_map)
        ts.compute_res()
        del ts

    print("Done. Time: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))
