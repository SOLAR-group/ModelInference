import datetime
import time

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.factory import get_decomposition
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling


# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEAD(GeneticAlgorithm):

    def __init__(self,
                 fsm,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        ref_dirs : {ref_dirs}

        decomposition : {{ 'auto', 'tchebi', 'pbi' }}
            The decomposition approach that should be used. If set to `auto` for two objectives `tchebi` and for more than
            two `pbi` will be used.

        n_neighbors : int
            Number of neighboring reference lines to be used for selection.

        prob_neighbor_mating : float
            Probability of selecting the parents in the neighborhood.


        """
        self.n_gen = 1
        self.fsm = fsm
        self.fsm.gen = self.n_gen
        self.infeasible_count = 0

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs
        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

    def _initialize(self):
        # self.start_time = time.perf_counter()

        if isinstance(self.decomposition, str):

            # set a string
            decomp = self.decomposition

            # for one or two objectives use tchebi otherwise pbi
            if decomp == 'auto':
                if self.problem.n_obj <= 2:
                    decomp = 'tchebi'
                else:
                    decomp = 'pbi'

            # set the decomposition object
            self._decomposition = get_decomposition(decomp)

        else:
            self._decomposition = self.decomposition

        super()._initialize()
        self.ideal_point = np.min(self.pop.get("F"), axis=0)
        self.nadir_point = np.max(self.pop.get("F"), axis=0)

    def _next(self):
        repair, crossover, mutation = self.mating.repair, self.mating.crossover, self.mating.mutation

        # retrieve the current population
        pop = self.pop
        self.fsm.gen = self.n_gen
        print("Gen:" + str(self.n_gen))
        # if not self.fsm.gen % 9:
        #     print("\nTime tot partial: " + str(
        #        datetime.timedelta(seconds=round(time.perf_counter() - self.start_time))) + "\n")

        time_cx = list()
        time_mut = list()
        # time_eval = list()
        # time_surv = list()
        # off_sizes = list()
        # iterate for each member of the population in random order
        for i in np.random.permutation(len(pop)):

            # all neighbors of this individual and corresponding weights
            N = self.neighbors[i, :]
            if np.random.random() < self.prob_neighbor_mating:
                parents = N[np.random.permutation(self.n_neighbors)][:crossover.n_parents]
            else:
                parents = np.random.permutation(self.pop_size)[:crossover.n_parents]

            # do recombination and create an offspring
            start_cx = time.perf_counter()
            off = crossover.do(self.problem, pop, parents[None, :])
            time_cx.append(time.perf_counter() - start_cx)
            start_mut = time.perf_counter()
            off = mutation.do(self.problem, off)
            time_mut.append(time.perf_counter() - start_mut)
            off = off[np.random.randint(0, len(off))]
            #off_sizes.append(len(off.X[0].States))

            # repair first in case it is necessary
            if repair:
                off = self.repair.do(self.problem, off, algorithm=self)

            #start_eval = time.perf_counter()
            # evaluate the offspring
            self.evaluator.eval(self.problem, off)
            #time_eval.append(time.perf_counter() - start_eval)

            if off.X[3] == "infeasible":
                self.infeasible_count += 1
                continue

            #start_surv = time.perf_counter()
            # update the ideal point and the nadir point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)
            self.nadir_point = np.max(np.vstack([self.nadir_point, off.F]), axis=0)

            # calculate the decomposed values for each neighbor
            FV = self._decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal_point, nadir_point=self.nadir_point)
            off_FV = self._decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal_point, nadir_point=self.nadir_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            pop[N[I]] = off
            #time_surv.append(time.perf_counter() - start_surv)

        print("Time crossover " + str(datetime.timedelta(seconds=round(sum(time_cx)))))
        print("Time mutation " + str(datetime.timedelta(seconds=round(sum(time_mut)))))
        # print("Offsprings size: ", end="")
        # for size in off_sizes:
        #     if size <= 500:
        #         print(size, end=" ")
        # print("")
        # print("Time evaluation: " + str(round(sum(time_eval), 1)))
        # print("Time survival " + str(round(sum(time_surv), 1)))
        # print("Population sizes after replacement: ", end="")
        # for indv in pop:
        #     print(len(indv.X[0].States), end=" ")
        # print("")
        if not self.n_gen % 9:
            print("Infeasible solutions: " + str(self.infeasible_count) + "\n")

parse_doc_string(MOEAD.__init__)