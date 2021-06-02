import os

import numpy as np
from pymoo.factory import get_performance_indicator
from pymoo.performance_indicator.hv import Hypervolume

import time
import pickle
import random
import datetime


class PerformanceIndicators:
    def __init__(self, prod, moeas):
        self.product = prod
        self.algorithms = moeas
        self.hv_bounds = list()
        self.n_run = 30

    def computeReferenceParetoFront(self):
        ref_pareto_front = list()
        for alg in self.algorithms:
            for run in range(1, self.n_run + 1):
                front_file = open('../../../data/' + product + '/' + alg + '/'
                                  + str(run) + '_EAfront' + alg + '_' + product + '.pkl', 'rb')
                pareto_set = pickle.load(front_file)
                front_file.close()
                for model in pareto_set:
                    ref_pareto_front.append(model.F)
        return np.asarray(ref_pareto_front)

    def computeHV(self, pareto_front, low_b, up_b):
        hv = Hypervolume(normalize=True, bounds=[low_b, up_b])
        hv_res = (hv.calc(pareto_front))
        return hv_res

    def computeIGD(self, pareto_front, ref_pareto_front, low_b, up_b):
        igd = get_performance_indicator("igd", ref_pareto_front, normalize=True, bounds=[low_b, up_b])
        igd_res = igd.calc(pareto_front)
        return igd_res

    def findBounds(self):
        min_UnRec, min_Size, min_UnObs = 100000, 100000, 100000
        max_UnRec, max_Size, max_UnObs = 0, 0, 0

        for alg in self.algorithms:
            for run in range(1, self.n_run + 1):
                front_file = open('../../../data/' + product + '/' + alg + '/'
                                  + str(run) + '_EAfront' + alg + '_' + product + '.pkl', 'rb')
                pareto_set = pickle.load(front_file)
                front_file.close()

                for indv_fit in pareto_set.get("F"):
                    if indv_fit[0] < min_UnRec:
                        min_UnRec = indv_fit[0]
                    if indv_fit[1] < min_Size:
                        min_Size = indv_fit[1]
                    if indv_fit[2] < min_UnObs:
                        min_UnObs = indv_fit[2]
                    if indv_fit[0] > max_UnRec:
                        max_UnRec = indv_fit[0]
                    if indv_fit[1] > max_Size:
                        max_Size = indv_fit[1]
                    if indv_fit[2] > max_UnObs:
                        max_UnObs = indv_fit[2]

        low_bound = np.asarray([min_UnRec, min_Size, min_UnObs])
        up_bound = np.asarray([max_UnRec, max_Size, max_UnObs])

        return low_bound, up_bound


if __name__ == '__main__':
    # prod = ["kate", "Vibe", "krita", "LibreOffice", "Firefox OS", "Firefox for Android", "SeaMonkey", "Thunderbird", "Calendar", "BIRT"]
    prod = ["BIRT"]
    bounds = {"kate": [[179, 1, 0], [401, 500, 500]],
              "Vibe": [[242, 1, 0], [425, 500, 500]],
              "krita": [[711, 1, 0], [958, 500, 500]],
              "LibreOffice": [[781, 1, 0], [1040, 501, 500]],
              "Firefox OS": [[849, 1, 0], [1082, 500, 500]],
              "Firefox for Android": [[799, 1, 0], [1054, 501, 500]],
              "SeaMonkey": [[1236, 1, 0], [1598, 500, 500]],
              "Thunderbird": [[1377, 1, 0], [1654, 500, 500]],
              "Calendar": [[2262, 1, 0], [2576, 501, 500]],
              "BIRT": [[3960, 1, 0], [4361, 501, 500]]}

    startf = time.perf_counter()
    n_run = 30
    algorithms = ['NSGAII', 'NSGAIII', 'MOEAD']
    hv_results = {}
    igd_results = {}
    os.makedirs(os.path.dirname('../../../data/Performance Indicator/'), exist_ok=True)
    for indx, product in enumerate(prod):
        # hv_file = open('../../../data/Performance Indicator/HV/hv_' + program + '.txt', 'w+')
        igd_file = open('../../../data/Performance Indicator/IGD/igd_' + product + '.txt', 'w+')
        igd_ref_file = open('../../../data/Performance Indicator/IGD/igd_ref_' + product + '.pkl', 'wb')
        # hv_file.write("Product: " + program + "\n")
        igd_file.write("Product: " + product + "\n")
        print("\n" + product + "\n")
        # pi = PerformanceIndicators(program, algorithms)
        # low_bound, up_bound = pi.findBounds() # Used only when need to compute HV bounds
        # print(low_bound, up_bound)
        pi = PerformanceIndicators(product, algorithms)
        reference_pareto = pi.computeReferenceParetoFront()
        pickle.dump(reference_pareto, igd_ref_file)
        igd_ref_file.close()
        igd_ref_file = open('../../../data/Performance Indicator/IGD/igd_ref_' + product + '.pkl', 'rb')
        reference_pareto = pickle.load(igd_ref_file)
        igd_ref_file.close()

        igd_results[product] = {}
        for alg in algorithms:
            igd_results[product][alg] = list()
            for run_num in range(1, n_run + 1):
                print("\rRun: " + str(run_num) + "        ", end="")
                front_file = open('../../../data/' + product + '/' + alg + '/'
                                  + str(run_num) + '_EAfront' + alg + '_' + product + '.pkl', 'rb')
                pareto_set = pickle.load(front_file)
                front_file.close()
                igd = pi.computeIGD(pareto_set.get("F"), reference_pareto, np.asarray(bounds[product][0]), np.asarray(bounds[product][1]))
                igd_results[product][alg].append(round(igd, 2))
            avg_sum = 0

            igd_file.write("\nAlgorithm: " + str(alg) + "\n")
            igd_file.write("IGD values (30 runs): ")
            for igd_value in igd_results[product][alg]:
                igd_file.write(str(igd_value) + ", ")
                avg_sum += igd_value
            igd_file.write("\nAverage IGD (30 runs): " + str(round(avg_sum/n_run, 2)) + "\n")

        igd_file.close()

        # hv_results[program] = {}
        # for alg in algorithms:
        #     print(alg)
        #     hv_results[program][alg] = list()
        #     for run_num in range(1, n_run + 1):
        #         print("\rRun: " + str(run_num) + "        ", end="")
        #         front_file = open('../../../data/' + program + '/' + alg + '/'
        #                           + str(run_num) + '_EAfront' + alg + '_' + program + '.pkl', 'rb')
        #         pareto_set = pickle.load(front_file)
        #         front_file.close()
        #         hv = pi.computeHV(pareto_set.get("F"), np.asarray(bounds[program][0]), np.asarray(bounds[program][1]))
        #         hv_results[program][alg].append(round(hv, 2))
        #     avg_sum = 0
        #
        #     hv_file.write("\nAlgorithm: " + str(alg) + "\n")
        #     hv_file.write("Hypervolume values (30 runs): ")
        #     for hv_value in hv_results[program][alg]:
        #         hv_file.write(str(hv_value) + ", ")
        #         avg_sum += hv_value
        #     hv_file.write("\nAverage Hypervolume (30 runs): " + str(round(avg_sum/n_run, 2)) + "\n")
        #
        # hv_file.close()

    print("Done. Time: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))
