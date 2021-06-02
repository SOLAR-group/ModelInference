import datetime
import pickle
import sys
import time

import numpy as np
from pymoo.factory import get_termination, get_reference_directions
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.termination.collection import TerminationCollection

from src.MOEA.pymoo.FSMop_pymoo import *
from src.MOEA.pymoo.MOEAD.PymooMOEADModified import MOEAD
from src.MOEA.pymoo.MOEAD.PymooTchebicheffModified import Tchebicheff


class EApymooMOEAD:
    def __init__(self, n_run, prod, step):
        self.n_run = n_run
        self.program = prod
        self.objectives = "UnRec-Size-UnObs"
        self.step_map_train = step
        self.fsm = FSMop(self.step_map_train)
        self.n_pop = 300
        # self.n_gen = 100
        self.n_eval = 26000
        self.fsm.MHPR = 0.55
        # self.n_offsprings = 259

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
        def __init__(self, fsm):
            super().__init__(n_var=4, n_obj=3, n_constr=0, elementwise_evaluation=True, type_var=np.object)
            self.fsm = fsm

        def _evaluate(self, x, out, *args, **kwargs):
            if x[3] == "feasible":
                f1 = self.fsm.UnRec(x)
                f2 = len(x[0].States)
                f3 = self.fsm.UnObs(x[0])
                if f3 > 10000:
                    x[3] = "infeasible"
                    f1 = f2 = f3 = 100000
                out["F"] = np.array([f1, f2, f3], dtype=np.int)
            else:
                f1 = f2 = f3 = 100000
                out["F"] = np.array([f1, f2, f3], dtype=np.int)

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

            # create a population object
            off = pop.new("X", X)

            return off

        def _do(self, problem, X, **kwargs):  # The input X has the following shape (n_parents, n_matings, n_var)
            n_parents, n_matings, n_var = X.shape
            Y = np.full((self.n_offsprings, n_matings, n_var), None)
            for k in range(n_matings):
                parent1, parent2 = X[0, k], X[1, k]
                offspring = self.fsm.crossoverM(parent1, parent2)
                if len(offspring[0].States) > 500:
                    offspring[3] = "infeasible"
                Y[0, k] = offspring

            return Y

    class MyMutation(Mutation):
        def __init__(self, fsm, start_t):
            super().__init__()
            self.fsm = fsm
            self.MUTPB = 0.55
            self.start_t = start_t

        def _do(self, problem, X, **kwargs):
            for i in range(len(X)):
                if random.random() <= self.MUTPB:
                    if X[i][3] == "feasible":
                        X[i] = self.fsm.mutationM(X[i])
                        if len(X[i][0].States) > 500:
                            X[i][3] = "infeasible"

            return X

    def run(self):
        flog = open('results/' + self.program + '/MOEAD/' + self.objectives + "/" + str(self.n_run) + '_LogN_' + self.program + '.txt', 'w', 1)
        sys.stdout = flog
        start = time.perf_counter()
        start_p = time.process_time()
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=23)
        decomposition = Tchebicheff()
        algorithm = MOEAD(fsm=self.fsm,
                          pop_size=self.n_pop,
                          n_neighbors=20,
                          ref_dirs=ref_dirs,
                          decomposition=decomposition,
                          prob_neighbor_mating=0.9,
                          sampling=self.MySampling(self.fsm, self.step_map_train),
                          crossover=self.MyCrossover(self.fsm),
                          mutation=self.MyMutation(self.fsm, start),
                          eliminate_duplicates=False)
        problem = self.MyProblem(self.fsm)
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

        f = open('results/' + self.program + '/MOEAD/' + self.objectives + "/" + str(self.n_run) + 'RunMOEAD' + self.program + '.txt', 'w')
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
            if fitness_values[i][2] > 500 or len(pareto_set[i].X[0].States) > 500:
                rem_indexes.append(i)
        pareto_set = np.delete(pareto_set, rem_indexes)

        # Update front
        I = NonDominatedSorting().do(pareto_set.get("F"), only_non_dominated_front=True)
        pareto_set = pareto_set[I]

        tot_p = round(time.process_time() - start_t)
        ff = open('results/' + self.program + '/MOEAD/' + self.objectives + "/" + str(self.n_run) + '_EAfrontMOEAD_' + self.program + '.pkl',
                  'wb')
        pickle.dump(pareto_set, ff)
        ff.close()

        f.write("\nLast min_hop: " + str(datetime.timedelta(seconds=round(tot_p))))
        f.write("\nTime tot (gen+min): " + str(round(time.process_time() - start_p)) + " : " + str(
            datetime.timedelta(seconds=round(time.process_time() - start_p))))
        f.close()
        sys.stdout = sys.__stdout__
        # plot = Scatter()
        # plot.add(res.F, color="red")
        # plot.show()

        del pareto_set, res, self.fsm, self.step_map_train
