from src.MOEA.deap.FSMop import *
from deap import creator, base, tools
import time
import sys
import pickle
import random
import datetime


class EAdeapGA:
    def __init__(self, n_run, prod, step):
        self.n_run = n_run
        self.product = prod
        self.step_map_train = step
        self.fsm = FSMop(self.step_map_train)
        self.counter = 0
        self.n_pop = 1000
        self.iter_conv = 0
        self.pf = []
        self.gn = 1
        self.flag_cv = False
        self.pareto = None
        self.MUTPB = 0.55
        self.min_obj = [float('Inf'), float('Inf')]
        self.start_g = 0
        self.cnt = 0

    def createIndiv(self):
        chooser = None
        if random.randint(3, 4) == 4:
            chooser = random.sample(range(len(self.step_map_train)), 4)
        else:
            chooser = random.sample(range(len(self.step_map_train)), 3)

        fsU = self.fsm.union(self.fsm.dfaList[chooser[0]].dup(), self.fsm.dfaList[chooser[1]].dup())
        fsU = self.fsm.union(fsU, self.fsm.dfaList[chooser[2]].dup())

        if len(chooser) == 3:
            self.rtr_sigma(fsU)
            fsU = fsU.minimalHopcroft()
            fsU.Sigma = self.fsm.sigma
            return fsU
        else:
            fsU = self.fsm.union(fsU, self.fsm.dfaList[chooser[3]].dup())
            self.rtr_sigma(fsU)
            fsU = fsU.minimalHopcroft()
            fsU.Sigma = self.fsm.sigma
            return fsU

    # Individual are made by single trace
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
        # After convergence or after a specific n. of generations,
        # I let him done some other iterations choosing the parents only from a set of the unique individual
        # and also the pareto
        if self.flag_cv or (95 <= self.gn <= 100):
            self.cnt += 1
            ls_v = self.uniq_ind(individuals)
            ln_par = 0
            if len(self.pareto.items) >= 70:
                ln_par = 70
            else:
                ln_par = len(self.pareto.items)
            if len(ls_v) + ln_par >= 10:
                for j in range(500):
                    rs = random.sample(range(0, len(ls_v) + ln_par), 2)
                    ind_t = []
                    for l in rs:
                        if l < len(ls_v):
                            ind_t.append(individuals[ls_v[l]])
                        else:
                            l -= len(ls_v)
                            ind_t.append(self.pareto.items[l])
                    chosen.append(tourn(ind_t[0], ind_t[1]))
                return chosen
        for i in range(0, len(individuals), 2):
            chosen.append(tourn(individuals[in_1[i]], individuals[in_1[i + 1]]))
        return chosen

    def find_min(self):
        min_val = list()
        min_val.append(self.pareto.keys[0].values[0])
        min_val.append(self.pareto.keys[0].values[1])

        for ft in self.pareto.keys:
            for o in range(len(ft.values) - 1):
                if ft.values[o] < min_val[o]:
                    min_val[o] = ft.values[o]
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
        flog = open('data/' + self.product + '/MHGA/' + str(self.n_run) + '_LogGA_' + self.product + '.txt', 'w', 1)
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

        start = time.perf_counter()

        pop = toolbox.population(self.n_pop)

        for i in range(len(pop)):
            pop[i].min_hop = False
            pop[i].rec_trace = set()

        print("Time create pop: " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

        self.pareto = tools.ParetoFront(similar=self.cmp_appr_dfa)

        start = time.perf_counter()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        self.fsm.MHPR = 0.55
        fl_conv = True
        print("Time fit child: %.2gs" % (time.perf_counter() - start))
        only_pa = 0
        startf = time.perf_counter()
        start_p = time.process_time()
        time_1cv_p = 0
        remove_cnt = 0
        for g in range(1, 101):

            if not g % 9:
                print(
                    "\nTime tot partial: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))) + "\n")
                print("Removed: "+str(remove_cnt))

            self.fsm.gen = self.gn = g

            print("Gen:" + str(g))

            # Select the next generation individuals
            offsprings = toolbox.select(pop, len(pop))

            child = list()

            start = time.perf_counter()
            # Apply crossover
            for ind1, ind2 in zip(offsprings[::2], offsprings[1::2]):
                child.append(toolbox.mate(ind1, ind2))
            cxmut_time = round(time.perf_counter() - start)
            print("Time crossover: " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

            rem = list()
            for j in range(len(child)):
                if len(child[j][0].States) > 500:
                    rem.append(j)
            rem.sort(reverse=True)
            for i in rem:
                del child[i]

            start = time.perf_counter()
            # Apply mutation on the children
            for mutant in child:
                if random.random() < self.MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            cxmut_time += round(time.perf_counter() - start)
            print("Time mutation: " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

            rem = list()
            for j in range(len(child)):
                if len(child[j][0].States) > 500:
                    rem.append(j)
            rem.sort(reverse=True)
            for i in rem:
                del child[i]

            start = time.perf_counter()
            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, child)
            for ind, fit in zip(child, fitnesses):
                ind.fitness.values = fit

            print("Time fit child: " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

            rem = list()
            for j in range(len(child)):
                if child[j].fitness.values[2] > 10000:
                    rem.append(j)
            rem.sort(reverse=True)
            for i in rem:
                del child[i]
            remove_cnt += len(rem)

            start = time.perf_counter()
            # Reinsertion strategy
            toAdd = list()
            for i in random.sample(range(len(child)), len(child)):
                for j in random.sample(range(len(pop)), len(pop)):
                    if child[i].fitness.dominates(pop[j].fitness):
                        toAdd.append(i)
                        del pop[j]
                        break

            for i in toAdd:
                pop.append(child[i])
            #print("Adedd: " + str(len(toAdd)))
            print("Time GA reinsertion: %.2gs" % (time.perf_counter() - start))

            start = time.perf_counter()
            self.pareto.update(child)
            only_pa += round(time.perf_counter() - start)
            #print("Time Pareto: " + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

            # try:
            #     print(self.pareto.keys[-1].values, self.pareto.keys[-2].values, self.pareto.keys[-3].values,
            #           self.pareto.keys[-4].values, self.pareto.keys[-5].values, self.pareto.keys[-6].values)
            # except IndexError:
            #     pass

        print("Stop generation")
        print("Time tot gen: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))

        f = open('data/' + self.product + '/MHGA/' + str(self.n_run) + 'RunGA' + self.product + '.txt', 'w')
        f.write(str(self.fsm.gen) + "\n" + str(datetime.timedelta(seconds=round(time.process_time() - start_p))))
        f.write("\nOnly Pareto fraction: " + str(datetime.timedelta(seconds=round(only_pa))))

        print("Init Pareto minimization")

        start_l = time.process_time()
        pop_p = list()
        pp_it = self.pareto.items
        del self.pareto
        mh_p = tools.ParetoFront(similar=self.cmp_appr_dfa)
        for i in range(len(pp_it)):
            if not pp_it[i].min_hop:
                self.rtr_sigma(pp_it[i][0])
                pp_it[i][0] = pp_it[i][0].minimalHopcroft()
                pp_it[i][0].Sigma = self.fsm.sigma
            pop_p.append(pp_it[i])

        fitnesses = toolbox.map(toolbox.evaluate, pop_p)
        for ind, fit in zip(pop_p, fitnesses):
            ind.fitness.values = fit

        rem = list()
        for j in range(len(pop_p)):
            if pop_p[j].fitness.values[2] > 500 or len(pop_p[j][0].States) > 500:
                rem.append(j)
        rem.sort(reverse=True)
        for h in rem:
            del pop_p[h]

        mh_p.update(pop_p)
        tot_p = round(time.process_time() - start_l)

        ff = open('data/' + self.product + '/MHGA/' + str(self.n_run) + '_EAfrontMHGA_' + self.product + '.pkl', 'wb')
        pickle.dump(mh_p, ff)
        ff.close()

        f.write("\nTime min_hop: " + str(datetime.timedelta(seconds=round(tot_p))))
        f.write("\nTime tot (gen+min): " + str(round(time.process_time() - start_p - time_1cv_p)) + " : " + str(
            datetime.timedelta(seconds=round(time.process_time() - start_p - time_1cv_p))))
        f.close()

        # print("Time tot " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))
        sys.stdout = sys.__stdout__
        del mh_p, pp_it, pop_p, self.fsm, self.pf, self.step_map_train
