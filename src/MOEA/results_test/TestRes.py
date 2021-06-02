from deap import tools, creator, base

from src.MOEA.deap.FSMop import *
import time
import pickle
import random
import datetime
import collections

'''
    Script used to test result
'''

class TestR:
    def __init__(self, prod, steps):
        self.product = prod
        self.step_map = steps
        #self.fsm = FSMop(self.step_map_train)

    def cmp_appr_dfa(self, dfa1, dfa2):
        dfa1, dfa2 = dfa1[0], dfa2[0]
        if len(dfa1.States) != len(dfa2.States):
            return False
        self.rtr_sigma(dfa2)
        self.rtr_sigma(dfa1)
        try:
            if (dfa1.Sigma != dfa2.Sigma) or (dfa1._uniqueStr() != dfa2._uniqueStr()):
                dfa1.Sigma = dfa2.Sigma = self.fsm.sigma
                return False
            else:
                dfa1.Sigma = dfa2.Sigma = self.fsm.sigma
                return True
        except ValueError:
            dfa1.Sigma = dfa2.Sigma = self.fsm.sigma
            return False

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
        #3 is the time, 4 is the number of solution, 5 is avg bug for pareto, 6 tot of bug, 7 traces, 8 traces x bug
        alg_tris = {"GA":[[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], 0, 0, 0, 0, 0, 0], "NSGAII": [[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], 0, 0, 0, 0, 0, 0]}
        trs = {"NSGAII": 0, "GA": 0}
        random.seed()

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
        print("Num bug trace test: " + str(len(self.step_map) - round((len(self.step_map) * 80) / 100)))
        start = time.perf_counter()
        sgm = fsm_t.sigma.union(fsm_r.sigma)

        iter_num = 30

        # ["NSGAIII", "NSGAII", "MHGA"]
        for alg in ["NSGAIII"]:
            all_sol = 0
            bug_pareto = 0
            bug_set = set()
            print("---")
            count_ga = 0
            for n_run in range(1, 31):
                print("\r" + str(n_run) + "        ", end="")
                bug_par = set()
                tim = 0
                front_file = open('data/' + product + '/'
                                  + alg + '/' + str(n_run) + '_EAfront' + alg + '_' + product + '.pkl', 'rb')
                pareto = pickle.load(front_file)
                front_file.close()

                for k in range(len(pareto.items)):
                    pareto.items[k][0].Sigma = sgm
                    j = 0
                    trs[alg] += pareto.items[k][0].countTransitions()
                    count_ga += fsm_t.traces(pareto.items[k][0])
                    for st in fsm_t.step_listSP:
                        if pareto.items[k][0].evalWordP(st, pareto.items[k][0].Initial):
                            bug_par.add(j)
                            bug_set.add(j)
                        j += 1
                bug_pareto += len(bug_par)
                # tim = int(linecache.getline(fold+program+'/'+alg+'/'+str(n_run)+'Run'+alg+program+'.txt', 3).split(':')[1]) used for NSGAII
                # tim = int(linecache.getline(fold+program+'/'+alg+'/'+str(n_run)+'Run'+alg+program+'.txt', 5).split(':')[1]) used for MHGA
                tim = 10
                alg_tris[alg][3] += tim
                all_sol += len(pareto.keys)
                for j in range(3):
                    for i in range(len(pareto.keys)):
                        if pareto.keys[i].values[j] < alg_tris[alg][j][0]:
                            alg_tris[alg][j][0] = pareto.keys[i].values[j]
                        if pareto.keys[i].values[j] >= alg_tris[alg][j][2]:
                            alg_tris[alg][j][2] = pareto.keys[i].values[j]
                        alg_tris[alg][j][1] += pareto.keys[i].values[j]
                del pareto
            for j in range(3):
                alg_tris[alg][j][1] = round(alg_tris[alg][j][1]/all_sol, 2)
            alg_tris[alg][3] = round(alg_tris[alg][3]/iter_num)
            alg_tris[alg][4] = round(all_sol/iter_num)
            alg_tris[alg][5] = round(bug_pareto/iter_num, 1)
            alg_tris[alg][6] = len(bug_set)
            alg_tris[alg][7] = round(count_ga/iter_num)
            alg_tris[alg][8] = round(alg_tris[alg][7]/alg_tris[alg][5])
            trs[alg] = round(trs[alg] / all_sol)

        print("\n"+str(alg_tris))
        print("\nTime: %.2gs" % (time.perf_counter() - start))
        print(str(trs))


if __name__ == '__main__':
    # Datasets are ordered by size
    # ["kate", "Vibe", "krita", "LibreOffice", "Firefox OS", "Firefox for Android", "SeaMonkey", "Thunderbird", "Calendar", "BIRT"]

    software = ["Thunderbird"]
    startf = time.perf_counter()
    step = list()
    for product in software:
        print(product + "\n")
        steps_file = open('data/bug/datasetF/BugStepsMappedPp_' + product + '_plus.pkl', 'rb')
        step_map = pickle.load(steps_file)
        steps_file.close()
        ts = TestR(product, step_map)
        ts.compute_res()
        del ts

    print("Done. Time: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))
