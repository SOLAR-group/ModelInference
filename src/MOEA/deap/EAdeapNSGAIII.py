from deap.tools import *
from src.MOEA.deap.FSMop import *
from deap import creator, base, tools
import time
import sys
import pickle
import random
import datetime
import copy


class EAdeapNSGAIII:
    def __init__(self, n_run, prod, step):
        self.n_run = n_run
        self.product = prod
        self.step_map_train = step
        self.fsm = FSMop(self.step_map_train)
        self.cpp = copy.deepcopy(self.fsm)
        self.counter = 0
        self.n_pop = 1000
        self.iter_conv = 0
        self.pf = []
        self.gn = 1
        self.flag_cv = False
        self.MUTPB = 0.55
        self.min_obj = [float('Inf'), float('Inf')]
        self.iter_max = 0
        self.time_exp = 64800 #18 hours
        self.start_g = 0
        #self.offsprings = None

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
            fsU = fsU.minimalHopcroft()
            fsU.Sigma = self.fsm.sigma
            return fsU
        else:
            self.rtr_sigma(fsU)
            fsU = fsU.minimalHopcroft()
            fsU.Sigma = self.fsm.sigma
            return fsU

    # Indivuduals made of a single trace
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

    def selTournamentD(self, individuals, k):

        def tourn(ind1, ind2):
            if ind1.fitness.dominates(ind2.fitness):
                return ind1
            elif ind2.fitness.dominates(ind1.fitness):
                return ind2

            if random.random() <= 0.5:
                return ind1
            return ind2

        in_1 = random.sample(range(len(individuals)), k)

        chosen = []
        for i in range(0, len(individuals), 2):
            chosen.append(tourn(individuals[in_1[i]], individuals[in_1[i + 1]]))
        return chosen

    def find_min(self, fr_pop):
        min_val = list()
        min_val.append(fr_pop[0].fitness.values[0])
        min_val.append(fr_pop[0].fitness.values[1])

        for ft in fr_pop:
            for o in range(len(ft.fitness.values)-1):
                if ft.fitness.values[o] < min_val[o]:
                    min_val[o] = ft.fitness.values[o]
        return min_val

    # Approx comparison, quicker
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

    def uniq_ind(self, pop):
        all_pop = set(list(range(self.n_pop)))
        ind_sg = set()
        flag = True
        while flag:
            to_remove = set()
            if len(all_pop):
                i = all_pop.pop()
                ind_sg.add(i)
                for j in all_pop:
                    if self.cmp_appr_dfa(pop[j], pop[i]):
                        to_remove.add(j)
                all_pop.difference_update(to_remove)
            else:
                flag = False
        return list(ind_sg)

    def rtr_sigma(self, dfa):
        sgm = set()
        for k in dfa.delta:
            for key in dfa.delta[k]:
                sgm.add(key)
        dfa.Sigma = sgm

    def compute(self):
        flog = open('data/'+self.product+'/NSGAIII/' + str(self.n_run) + '_LogN_' + self.product + '.txt', 'w', 1)
        sys.stdout = flog
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("cdfa", self.createIndiv)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.cdfa, 1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluateI)
        toolbox.register("mate", self.fsm.crossoverM)
        toolbox.register("mutate", self.fsm.mutationM)
        toolbox.register("select", self.selTournamentD)
        toolbox.register("map", map)
        ref_points = uniform_reference_points(nobj=3, p=43)
        toolbox.register("selectNSGA3", selNSGA3WithMemory(ref_points))

        start = time.perf_counter()

        pop = toolbox.population(self.n_pop)

        for i in range(len(pop)):
            pop[i].min_hop = False
            pop[i].rec_trace = set()

        print("Time create pop " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

        start = time.perf_counter()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("Time fit child: %.2gs" % (time.perf_counter() - start))
        self.fsm.MHPR = 0.55
        self.start_g = startf = time.perf_counter()
        start_p = time.process_time()
        time_cent = 0
        remove_cnt = 0
        for g in range(1, 101):

            if not g % 9:
                print("\nTime tot partial: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))) + "\n")
                print("Removed: "+str(remove_cnt))

            self.fsm.gen = self.gn = g
            print("Gen:" + str(g))

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))

            child = list()

            start = time.perf_counter()
            # Apply crossover on the offspring
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                child.append(toolbox.mate(ind1, ind2))
            cxmut_time = round(time.perf_counter() - start)
            print("Time crossover " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

            rem = list()
            for j in range(len(child)):
                if len(child[j][0].States) > 500:
                    rem.append(j)
            rem.sort(reverse=True)
            for i in rem:
                del child[i]
            remove_cnt += len(rem)

            start = time.perf_counter()
            # Apply mutation on the offspring
            for mutant in child:
                if random.random() <= self.MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            cxmut_time += round(time.perf_counter() - start)
            print("Time mutation " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

            start = time.perf_counter()
            rem = list()
            for j in range(len(child)):
                if len(child[j][0].States) > 500:
                    rem.append(j)
            rem.sort(reverse=True)
            for i in rem:
                del child[i]
            remove_cnt += len(rem)

            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, child)
            for ind, fit in zip(child, fitnesses):
                ind.fitness.values = fit

            rem = list()
            for j in range(len(child)):
                if child[j].fitness.values[2] > 10000:
                    rem.append(j)
            rem.sort(reverse=True)
            for i in rem:
                del child[i]
            remove_cnt += len(rem)
            print("Time fit child: " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

            start = time.perf_counter()
            all_p = pop + child

            pop = toolbox.selectNSGA3(individuals=all_p, k=self.n_pop)
            if (time.perf_counter() - startf) >= self.time_exp:
                break

        print("Stop generation")
        print("Time tot gen: " +
              str(round(time.process_time() - start_p)) +
              " : " +
              str(datetime.timedelta(seconds=round(time.process_time() - start_p))))
        ls_p = self.uniq_ind(pop)
        pop = [pop[i] for i in ls_p]

        # ff = open('data/' + self.program + '/NSGAIII/' + str(self.n_run) + '_EApopNSGAIII ' + self.program + '.pkl', 'wb')
        # pickle.dump(pop, ff)
        # ff.close()

        f = open('data/' + self.product + '/NSGAIII/' + str(self.n_run) + 'RunNSGAIII' + self.product + '.txt', 'w')
        f.write(str(self.fsm.gen) + ": " + str(round(time.process_time() - start_p))+" : "+str(datetime.timedelta(seconds=round(time.process_time() - start_p))))

        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        start_t = time.process_time()
        mh_p = tools.ParetoFront(similar=self.cmp_appr_dfa)

        print("Init pareto minimization")

        for i in range(len(front)):
            if not front[i].min_hop:
                self.rtr_sigma(front[i][0])
                front[i][0] = front[i][0].minimalHopcroft()
                front[i][0].Sigma = self.fsm.sigma

        fitnesses = toolbox.map(toolbox.evaluate, front)
        for ind, fit in zip(front, fitnesses):
            ind.fitness.values = fit

        rem = list()
        for j in range(len(front)):
            if front[j].fitness.values[2] > 500 or len(front[j][0].States) > 500:
                rem.append(j)
        rem.sort(reverse=True)
        for h in rem:
            del front[h]

        mh_p.update(front)
        tot_p = round(time.process_time() - start_t)

        ff = open('data/' + self.product + '/NSGAIII/' + str(self.n_run) + '_EAfrontNSGAIII_' + self.product + '.pkl',
                  'wb')
        pickle.dump(mh_p, ff)
        ff.close()

        f.write("\nLast min_hop: " + str(datetime.timedelta(seconds=round(tot_p))))
        f.write("\nTime tot (gen+min): " + str(round(time.process_time() - start_p)) + " : " + str(
                    datetime.timedelta(seconds=round(time.process_time() - start_p))))
        f.close()
        sys.stdout = sys.__stdout__
        del mh_p, front, pop, self.fsm, self.cpp, self.pf, self.step_map_train

