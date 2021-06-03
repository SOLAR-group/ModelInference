import linecache

import time
import pickle
import datetime
import collections

from src.MOEA.pymoo.FSMop_pymoo import FSMop

'''
    Script used to test result
'''


# TODO: add min and max for #Edges, Running Time and #Solutions

class TestR:
    def __init__(self, prod, steps):
        self.product = prod
        self.step_map = steps

    def compute_res(self):
        # 3 is the time, 4 is the number of solutions, 5 is avg bug for pareto, 6 tot of bug, 7 traces, 8 traces x bug
        # "GA": [[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], 0, 0, 0, 0, 0, 0],
        alg_tris = {"NSGAII": [[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0],
                               [float('Inf'), 0, 0], 0, 0, 0, 0],
                    "NSGAIII": [[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0],
                                [float('Inf'), 0, 0], 0, 0, 0, 0],
                    "MOEAD": [[float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0], [float('Inf'), 0, 0],
                              [float('Inf'), 0, 0], 0, 0, 0, 0]}
        trs = {"NSGAII": [float('Inf'), 0, 0], "NSGAIII": [float('Inf'), 0, 0], "MOEAD": [float('Inf'), 0, 0]}

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

        for alg in ["NSGAII", "NSGAIII", "MOEAD"]:
            all_sol = 0
            bug_pareto = 0
            print("---")
            count_ga = 0
            total_bugs_revealed = set()
            for n_run in range(1, iter_num + 1):
                print("\r" + str(n_run) + "        ", end="")
                running_time = 0
                front_file = open('../../../results/' + product + '/' + alg + '/UnRec-Size-UnObs/'
                                  + str(n_run) + '_EAfront' + alg + '_' + product + '.pkl', 'rb')
                pareto = pickle.load(front_file)
                front_file.close()
                bugs_revealed = set()

                for k in range(len(pareto)):
                    pareto[k].X[0].Sigma = sgm
                    j = 0
                    edges = pareto[k].X[0].countTransitions()
                    # Min #Edges
                    if edges < trs[alg][0]:
                        trs[alg][0] = edges
                    # Max #Edges
                    if edges > trs[alg][2]:
                        trs[alg][2] = edges

                    trs[alg][1] += edges
                    count_ga += fsm_t.traces(pareto[k].X[0])
                    # for bug in fsm_t.bugs_revealed(pareto[k].X[0]):
                    #    bugs_revealed.add(bug)
                    #    total_bugs_revealed.add(bug)
                bug_pareto += len(bugs_revealed)

                running_time = int(linecache.getline(
                    '../../../results/' + product + '/' + alg + '/UnRec-Size-UnObs/' + str(
                        n_run) + 'Run' + alg + product + '.txt', 3).split(':')[1])

                # Avg time
                alg_tris[alg][3][1] += running_time
                # Min time
                if running_time < alg_tris[alg][3][0]:
                    alg_tris[alg][3][0] = running_time
                # Max time
                if running_time > alg_tris[alg][3][2]:
                    alg_tris[alg][3][2] = running_time

                # Min #solutions
                if len(pareto) < alg_tris[alg][4][0]:
                    alg_tris[alg][4][0] = len(pareto)
                # Max #solutions
                if len(pareto) > alg_tris[alg][4][2]:
                    alg_tris[alg][4][2] = len(pareto)

                all_sol += len(pareto)
                for j in range(3):
                    for i in range(len(pareto)):
                        if pareto[i].F[j] < alg_tris[alg][j][0]:
                            alg_tris[alg][j][0] = pareto[i].F[j]
                        if pareto[i].F[j] >= alg_tris[alg][j][2]:
                            alg_tris[alg][j][2] = pareto[i].F[j]
                        alg_tris[alg][j][1] += pareto[i].F[j]
                del pareto
            for j in range(3):
                alg_tris[alg][j][1] = round(alg_tris[alg][j][1] / all_sol, 2)
            alg_tris[alg][3][1] = round(alg_tris[alg][3][1] / iter_num)
            alg_tris[alg][4][1] = round(all_sol / iter_num)
            # alg_tris[alg][5] = round(bug_pareto/iter_num, 1)
            # alg_tris[alg][6] = len(total_bugs_revealed)
            # alg_tris[alg][7] = round(count_ga/iter_num)
            # alg_tris[alg][8] = round(alg_tris[alg][7]/alg_tris[alg][5])
            trs[alg][1] = round(trs[alg][1] / all_sol)

            print(alg)
            # print("\nComputation Time: %.2gs" % (time.perf_counter() - start))
            print("Avg Number of Edges: " + str(trs[alg][1]))
            print("Min Number of Edges: " + str(trs[alg][0]))
            print("Max Number of Edges: " + str(trs[alg][2]))
            print("Avg Running Time: " + str(alg_tris[alg][3][1]))
            print("Min Running Time: " + str(alg_tris[alg][3][0]))
            print("Max Running Time: " + str(alg_tris[alg][3][2]))
            print("Avg Number of solutions: " + str(alg_tris[alg][4][1]))
            print("Min Number of solutions: " + str(alg_tris[alg][4][0]))
            print("Max Number of solutions: " + str(alg_tris[alg][4][2]))

            # print("avg bug for pareto: " + str(alg_tris[alg][5]))
            # print("tot of bug: " + str(alg_tris[alg][6]))
            # print("Avg Traces: " + str(alg_tris[alg][7]))
            # print("Traces per bug: " + str(alg_tris[alg][8]))
            # print("\n" + str(alg_tris))
            # print(str(trs))


if __name__ == '__main__':
    # Datasets are ordered by size
    # ["kate", "Vibe", "krita", "LibreOffice", "Firefox_OS",
    # "Firefox_for_Android", "SeaMonkey", "Thunderbird", "Calendar", "BIRT"]

    software = ["BIRT"]
    startf = time.perf_counter()
    step = list()
    for product in software:
        print(product + "\n")
        steps_file = open('../../../data/bug/dataset/BugStepsMappedPp_' + product + '_plus.pkl', 'rb')
        step_map = pickle.load(steps_file)
        steps_file.close()
        ts = TestR(product, step_map)
        ts.compute_res()
        del ts

    print("Done. Time: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))
