import datetime
import pickle
import sys
import time

import numpy as np
from pymoo.factory import get_termination
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.termination.collection import TerminationCollection

from src.MOEA.pymoo.FSMop_pymoo import *
from src.MOEA.pymoo.NSGA2.PymooNsga2Modified import NSGA2


class EApymooNSGAII:
    def __init__(self, n_run, prod, step, objectives):
        self.n_run = n_run
        self.program = prod
        self.step_map_train = step
        self.objectives = objectives
        self.fsm = FSMop(self.step_map_train)
        self.n_pop = 1000
        self.n_gen = 100
        self.n_eval = 26000
        self.fsm.MHPR = 0.55
        self.n_offsprings = 250

    def rtr_sigma(self, dfa):
        sgm = set()
        for k in dfa.delta:
            for key in dfa.delta[k]:
                sgm.add(key)
        dfa.Sigma = sgm

    def cmp_appr_dfa(self, dfa1, dfa2):
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

    def uniq_ind(self, pop):
        all_pop = set(list(range(len(pop))))
        ind_sg = set()
        flag = True
        while flag:
            to_remove = set()
            if len(all_pop):
                i = all_pop.pop()
                ind_sg.add(i)
                for j in all_pop:
                    if self.cmp_appr_dfa(pop[j][0], pop[i][0]):
                        to_remove.add(j)
                all_pop.difference_update(to_remove)
            else:
                flag = False
        return list(ind_sg)

    class MyProblem(Problem):
        def __init__(self, objectives, fsm):
            self.objectives = objectives
            self.fsm = fsm
            n_obj = 0
            if self.objectives == "UnRec-Size-UnObs":
                n_obj = 3
            elif self.objectives == "UnRec-Size" or self.objectives == "UnObs-Size" or self.objectives == "UnRec-UnObs":
                n_obj = 2
            else:
                n_obj = 1
            super().__init__(n_var=4, n_obj=n_obj, n_constr=0, elementwise_evaluation=True, type_var=np.object)
            print("Objectives: " + self.objectives)

        def _evaluate(self, x, out, *args, **kwargs):
            if x[3] == "feasible":
                if self.objectives == "UnRec-Size-UnObs":
                    f1 = self.fsm.UnRec(x)
                    f2 = len(x[0].States)
                    f3 = self.fsm.UnObs(x[0])
                    if f3 > 10000:
                        x[3] = "infeasible"
                        f1 = f2 = f3 = 100000
                    out["F"] = np.array([f1, f2, f3], dtype=np.int)
                elif self.objectives == "UnRec-Size":
                    f1 = self.fsm.UnRec(x)
                    f2 = len(x[0].States)
                    out["F"] = np.array([f1, f2], dtype=np.int)
                elif self.objectives == "UnObs-Size":
                    f1 = self.fsm.UnObs(x[0])
                    f2 = len(x[0].States)
                    if f1 > 10000:
                        x[3] = "infeasible"
                        f1 = f2 = 100000
                    out["F"] = np.array([f1, f2], dtype=np.int)
                elif self.objectives == "UnRec-UnObs":
                    f1 = self.fsm.UnRec(x)
                    f2 = self.fsm.UnObs(x[0])
                    if f2 > 10000:
                        x[3] = "infeasible"
                        f1 = f2 = 100000
                    out["F"] = np.array([f1, f2], dtype=np.int)
            else:
                if self.objectives == "UnRec-Size-UnObs":
                    f1 = f2 = f3 = 100000
                    out["F"] = np.array([f1, f2, f3], dtype=np.int)
                elif self.objectives == "UnRec-Size" or self.objectives == "UnObs-Size" or self.objectives == "UnRec-UnObs":
                    f1 = f2 = 100000
                    out["F"] = np.array([f1, f2], dtype=np.int)

    class MySampling(Sampling):
        def __init__(self, fsm, steps):
            self.step_map_train = steps
            self.fsm = fsm

        def rtr_sigma(self, dfa):
            sgm = set()
            for k in dfa.delta:
                for key in dfa.delta[k]:
                    sgm.add(key)
            dfa.Sigma = sgm

        def createIndiv(self):
            chooser = None
            if random.randint(2, 3) == 3:
                chooser = random.sample(range(len(self.step_map_train)), 3)
            else:
                chooser = random.sample(range(len(self.step_map_train)), 2)

            fsU = self.fsm.union(self.fsm.dfaList[chooser[0]].dup(), self.fsm.dfaList[chooser[1]].dup())

            if len(chooser) == 3:
                fsU = self.fsm.union(fsU, self.fsm.dfaList[chooser[2]].dup())

            self.rtr_sigma(fsU)
            fsU = fsU.minimalIncremental()
            fsU.Sigma = self.fsm.sigma
            return fsU

        def _do(self, problem, n_samples, **kwargs):
            start = time.perf_counter()
            X = np.full((n_samples, 4), None, dtype=np.object)

            for i in range(n_samples):
                X[i, 0] = self.createIndiv()
                X[i, 1] = set()  # rec_trace
                X[i, 2] = False  # min_hop
                X[i, 3] = "feasible"
            print("Gen:1")
            print("Time create pop " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))
            return X

    class MyCrossover(Crossover):
        def __init__(self, fsm):
            super().__init__(2, 1, prob=1)
            self.fsm = fsm
            self.gen = 2
            self.fsm.gen = self.gen

        def do(self, problem, pop, parents, **kwargs):
            if self.n_parents != parents.shape[1]:
                raise ValueError('Exception during crossover: Number of parents differs from defined at crossover.')

            # get the design space matrix form the population and parents
            X = pop.get("X")[parents.T].copy()

            # now apply the crossover probability
            do_crossover = np.random.random(len(parents)) < self.prob

            # execute the crossover
            _X = self._do(problem, X, **kwargs)

            X[:, do_crossover, :] = _X[:, do_crossover, :]

            # flatten the array to become a 2d-array
            X = X.reshape(-1, X.shape[-1])
            # take the first half to avoid duplicates
            X = X[0:250]
            # create a population object
            off = pop.new("X", X)

            return off

        def _do(self, problem, X, **kwargs):  # The input X has the following shape (n_parents, n_matings, n_var)
            start = time.perf_counter()
            print("Gen:" + str(self.gen))
            self.fsm.gen = self.gen
            self.gen += 1
            n_parents, n_matings, n_var = X.shape
            Y = np.full((self.n_offsprings, n_matings, n_var), None)

            for k in range(n_matings):
                parent1, parent2 = X[0, k], X[1, k]
                offspring = self.fsm.crossoverM(parent1, parent2)
                if len(offspring[0].States) > 500:
                    offspring[3] = "infeasible"
                Y[0, k] = offspring

            print("Time crossover " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))
            return Y

    class MyMutation(Mutation):
        def __init__(self, fsm, start_t):
            super().__init__()
            self.fsm = fsm
            self.MUTPB = 0.55
            self.start_t = start_t

        def _do(self, problem, X, **kwargs):
            start = time.perf_counter()
            for i in range(len(X)):
                if random.random() <= self.MUTPB:
                    if X[i][3] == "feasible":
                        X[i] = self.fsm.mutationM(X[i])
                        if len(X[i][0].States) > 500:
                            X[i][3] = "infeasible"

            print("Time mutation " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))
            # print("Offsprings size: ", end="")
            # for off in X:
            #     if off[3] == "feasible":
            #         print(len(off[0].States), end=" ")
            # print("")

            if not self.fsm.gen % 9:
                print("\nTime tot partial: " + str(
                    datetime.timedelta(seconds=round(time.perf_counter() - self.start_t))) + "\n")

            return X

    def run(self):

        flog = open('results/' + self.program + '/NSGAII/' + self.objectives + "/" + str(self.n_run) + '_LogN_'
                    + self.program + '.txt', 'w', 1)
        sys.stdout = flog
        start = time.perf_counter()
        start_p = time.process_time()
        algorithm = NSGA2(pop_size=self.n_pop,
                          sampling=self.MySampling(self.fsm, self.step_map_train),
                          crossover=self.MyCrossover(self.fsm),
                          mutation=self.MyMutation(self.fsm, start),
                          eliminate_duplicates=False,
                          n_offsprings=self.n_offsprings)

        problem = self.MyProblem(self.objectives, self.fsm)
        termination = TerminationCollection(get_termination("n_eval", self.n_eval), get_termination("time", "18:00:00"))
        res = minimize(problem, algorithm, termination, verbose=False)
        print("Stop generation")
        print("Time tot gen: " + str(round(time.process_time() - start_p)) + " : " + str(
            datetime.timedelta(seconds=round(time.process_time() - start_p))))

        # Removing duplicates from final population
        ls_p = self.uniq_ind(res.pop.get("X"))
        pareto_set = res.pop[ls_p]

        I = NonDominatedSorting().do(pareto_set.get("F"), only_non_dominated_front=True)
        pareto_set = pareto_set[I]

        f = open('results/' + self.program + '/NSGAII/' + self.objectives + "/" + str(self.n_run) + 'RunNSGAII'
                 + self.program + '.txt', 'w')
        f.write(str(res.algorithm.n_gen) + ": " + str(round(time.process_time() - start_p)) + " : " + str(
            datetime.timedelta(seconds=round(time.process_time() - start_p))))

        start_t = time.process_time()
        print("Init pareto minimization")
        for i in range(len(pareto_set)):
            if not pareto_set[i].X[2]:
                self.rtr_sigma(pareto_set[i].X[0])
                pareto_set[i].X[0] = pareto_set[i].X[0].minimalIncremental()
                pareto_set[i].X[0].Sigma = self.fsm.sigma

        res.algorithm.evaluator.eval(res.problem, pareto_set)

        # Removing individuals that exceed the threshold of 500 UnObs or Size
        rem_indexes = list()
        fitness_values = pareto_set.get("F")
        for i in range(len(fitness_values)):
            if self.objectives != "UnRec-Size":
                index = -1
                if self.objectives == "UnRec-Size-UnObs":
                    index = 2
                elif self.objectives == "UnObs-Size":
                    index = 0
                elif self.objectives == "UnRec-UnObs":
                    index = 1
                if fitness_values[i][index] > 500 or len(pareto_set[i].X[0].States) > 500:
                    rem_indexes.append(i)
            else:
                if len(pareto_set[i].X[0].States) > 500:
                    rem_indexes.append(i)
        pareto_set = np.delete(pareto_set, rem_indexes)

        # Update front
        I = NonDominatedSorting().do(pareto_set.get("F"), only_non_dominated_front=True)
        pareto_set = pareto_set[I]

        tot_p = round(time.process_time() - start_t)

        ff = open('results/' + self.program + '/NSGAII/' + self.objectives + "/" + str(self.n_run) + '_EAfrontNSGAII_'
                  + self.program + '.pkl',
                  'wb')
        pickle.dump(pareto_set, ff)
        ff.close()

        f.write("\nLast min_hop: " + str(datetime.timedelta(seconds=round(tot_p))))
        f.write("\nTime tot (gen+min): " + str(round(time.process_time() - start_p)) + " : " + str(
            datetime.timedelta(seconds=round(time.process_time() - start_p))))
        f.close()
        sys.stdout = sys.__stdout__

        del pareto_set, res, self.fsm, self.step_map_train
