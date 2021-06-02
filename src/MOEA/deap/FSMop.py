from FAdo.fa import *
import random
import copy


class FSMop:

    def __init__(self, step_map):
        self.step_list = []
        self.step_listSP = []
        self.sigma = set()
        self.cntUnObs1 = 0
        for key in step_map:
            self.step_list.append(step_map[key])
            tmp = []
            for i in range(0, len(step_map[key])):
                self.sigma.add(step_map[key][i]+" ")
                tmp.append(step_map[key][i]+" ")
            self.step_listSP.append(tmp)

        self.dfaList = [DFA() for i in range(len(self.step_list))]
        #self.dfaListBack = None
        self.buildDFA()
        self.gen = None
        self.MHPR = 0.6

    def buildDFA(self):
        j = 0
        for key in range(0, len(self.step_list)):
            self.dfaList[j].setSigma(self.sigma)
            for i in range(0, len(self.step_listSP[key]) + 1):
                self.dfaList[j].addState(str(i))
            self.dfaList[j].setInitial(0)
            for i in range(0, len(self.step_listSP[key])):
                self.dfaList[j].addTransition(i, self.step_listSP[key][i], i + 1)
                self.dfaList[j].addFinal(i)
            self.dfaList[j].addFinal(len(self.step_listSP[key]))
            j += 1
        return self.dfaList

    def UnObs(self, indv):
        lsEnum = self.lsenum(indv, indv.Initial)
        cntUnObs = 0
        for i in range(len(lsEnum)):
            flag = False
            obs = lsEnum[i].rstrip(" ").split(" ")
            for j in range(len(self.step_list)):
                flag = False
                if len(self.step_list[j]) < len(obs):
                    flag = True
                else:
                    for t in range(len(obs)):
                        if self.step_list[j][t] != obs[t]:
                            flag = True
                            break
                if not flag:
                    break
            if flag:
                cntUnObs += 1
        return cntUnObs

    def UnRec(self, indv):
        cntUnRec = 0
        j = 0
        rec = set()
        for st in self.step_listSP:
            try:
                if not indv[0].evalWordP(st, indv[0].Initial):
                    cntUnRec += 1
                else:
                    rec.add(j)
            except DFAsymbolUnknown:
                cntUnRec += 1
            j += 1
        #indv.rec_trace = rec
        return cntUnRec

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
                dfa1.Sigma = dfa2.Sigma = self.sigma
                return False
            else:
                dfa1.Sigma = dfa2.Sigma = self.sigma
                return True
        except ValueError:
            dfa1.Sigma = dfa2.Sigma = self.sigma
            return False

    def crossoverM(self, indv1, indv2):
        indv1 = copy.deepcopy(indv1)
        in_rec = indv2.rec_trace
        indv2 = indv2[0].dup()
        nDFA = indv1[0]
        if random.random() <= 0.5:
            if not self.cmp_appr_dfa(indv1[0], indv2):
                nDFA = self.union(indv1[0], indv2)
                indv1.rec_trace = indv1.rec_trace.union(in_rec)
                indv1.min_hop = False
            else:
                indv1 = self.addTraceM(indv1)
                nDFA = indv1[0]
                indv1.min_hop = False

        else:
            if not self.cmp_appr_dfa(indv1[0], indv2):
                nDFA = self.intersection(indv1[0], indv2)
                indv1.rec_trace = indv1.rec_trace.intersection(in_rec)
                indv1.min_hop = False
            else:
                nDFA = self.mergeRandK(indv1[0])
                indv1.min_hop = False
        indv1[0] = nDFA
        del indv1.fitness.values
        return indv1

    def union(self, indv_1, indv_2):
        indv = self.mergeUn(indv_1.toNFA(), indv_2.toNFA())
        #fs.makePNG(fileName="fsmUnm")
        return indv

    def mergeUn(self, indv1, indv2):
        ln1 = len(indv1.States)
        # Rewriting node name (just for precaution) and also append the new one
        for i in range(ln1 + len(indv2.States) - 1):
            if i < ln1:
                indv1.States[i] = str(i)
            else:
                indv1.States.append(str(i))

        [init1] = indv1.Initial
        [init2] = indv2.Initial

        ndMap = list(range(len(indv2.States)))
        self.nodeIDmap(ndMap, ln1, init1, init2)

        dictInit2 = indv2.delta.pop(init2)

        for state, to in indv2.delta.items():
            for a, [s] in to.items():
                indv2.delta[state][a] = {ndMap[s]}
            indv1.delta[ndMap[state]] = indv2.delta[state]

        for a, [s] in dictInit2.items():
            try:
                indv1.delta[init1][a] = {indv1.delta[init1][a].pop(), ndMap[s]}
            except:
                indv1.delta[init1][a] = {ndMap[s]}

        indv1.setFinal(indv1.indexList(indv1.States))
        return indv1.toDFA()

    def nodeIDmap(self, ndMap, ln, init1, init2):
        for i in range(len(ndMap)):
            if i < init2:
                ndMap[i] = i + ln
            elif i > init2:
                ndMap[i] = i + ln - 1
            else:
                ndMap[i] = init1

    def intersection(self, indv1, indv2):
        indv1.delFinals()
        indv2.delFinals()
        nDFA = DFA()
        tranS = set()
        stat = set()
        rn = random.random()
        if rn <= 0.5:
            self.dfs_recur(stat, tranS, indv1, indv2, indv1.Initial, indv2.Initial)
        else:
            self.dfs_recur(stat, tranS, indv2, indv1, indv2.Initial, indv1.Initial)
        if len(tranS):
            if rn <= 0.5:
                nDFA = self.creatNdfa(nDFA, tranS, stat, indv1.Initial)
            else:
                nDFA = self.creatNdfa(nDFA, tranS, stat, indv2.Initial)
            #nDFA.makePNG(fileName="fsmInt")
            return nDFA
        else:
            indv1.setFinal(indv1.indexList(indv1.States))
            indv2.setFinal(indv2.indexList(indv2.States))
            if random.random() <= 0.5:
                return self.mergeRandK(indv1)
            else:
                return self.mergeRandK(indv2)

    def dfs_recur(self, stat, tranS, indv1, indv2, v1, v2):
        indv1.addFinal(v1)
        indv2.addFinal(v2)
        str_ch2 = list(indv2.inputS(v2))

        for w in list(indv1.inputS(v1)):
            if w in str_ch2:
                v1nx = indv1.evalSymbol(v1, w)
                v2nx = indv2.evalSymbol(v2, w)
                if v1 == v1nx and v2 == v2nx:
                    stat.update([v1, v1nx])
                    tranS.add((v1, w, v1nx))
                    continue
                elif (indv1.finalP(v1nx) and v1 != v1nx) or (indv2.finalP(v2nx) and v2 != v2nx):
                    stat.update([v1, v1nx])
                    tranS.add((v1, w, v1nx))
                    continue
                else:
                    stat.update([v1, v1nx])
                    tranS.add((v1, w, v1nx))
                    self.dfs_recur(stat, tranS, indv1, indv2, v1nx, v2nx)

    def creatNdfa(self, nDFA, tranS, stat, init):
        nDFA.setSigma(self.sigma)
        tmpI = {}
        cnt = 0
        for s in stat:
            nDFA.addState(s)
            tmpI[s] = cnt
            cnt += 1
        nDFA.setInitial(tmpI[init])
        for tr in tranS:
            nDFA.addTransition(tmpI[tr[0]], tr[1], tmpI[tr[2]])
        nDFA.setFinal(nDFA.indexList(nDFA.States))
        return nDFA

    def addTraceM(self, indv):
        try:
            rn = random.sample(set(list(range(0, len(self.dfaList)))).difference(indv.rec_trace), 1)
        except ValueError:
            return indv
        indv[0] = self.mergeUn(indv[0].toNFA(), self.dfaList[rn[0]].dup().toNFA())
        indv.rec_trace.add(rn[0])
        return indv

    def mutationM(self, indv):
        if random.random() <= 0.5:
            indv = self.addTraceM(indv)
        else:
            if random.random() <= 0.9:
                indv[0] = self.mergeRandK(indv[0])
                if self.gen >= 5:
                    indv[0] = self.mergeRandK(indv[0], True)
            else:
                try:
                    rn = random.sample(range(len(indv[0].States)), 2)
                except ValueError:
                    return indv
                indv[0] = self.mergeRandSub(indv[0], rn[0], rn[1])

        if random.random() <= self.MHPR and len(indv[0].States) < 1000:
            self.rtr_sigma(indv[0])
            indv[0] = indv[0].minimalHopcroft()
            indv[0].Sigma = self.sigma
            indv.min_hop = True
        else:
            indv.min_hop = False
        return indv

    def mergeRandSub(self, indv, f, t):
        ts = {t}
        fs = {f}
        nSt = indv.addState()
        indv = indv.toNFA()
        nSts = {nSt}
        tranS = set()

        if f is not t:
            for state, to in indv.delta.items():
                    for a, s in to.items():
                        if f == state or t == state:
                            if s == fs or s == ts:
                                tranS.add((nSt, a, nSt))
                            else:
                                tranS.add((nSt, a, list(s)[0]))
                        elif fs == s or ts == s:
                            indv.delta[state][a] = nSts

        for tr in tranS:
            indv.addTransition(tr[0], tr[1], tr[2])

        if indv.initialP(t) or indv.initialP(f):
            indv.setInitial(nSts)
        indv.setFinal(indv.indexList(indv.States))
        dels = [t, f]
        indv.deleteStates(dels)
        indv.setSigma(self.sigma)
        indv = indv.toDFA()
        return indv

    def mergeRandK(self, indv, fl=False):
        lsind = list(range(len(indv.States)))
        if not fl:
            lsind.remove(indv.Initial)
        pop = random.sample(lsind, len(lsind))
        check = False
        for i in pop:
            lsind.remove(i)
            for j in random.sample(lsind, len(lsind)):
                if j != i:
                    ls11 = self.k2tail(indv, i)
                    ls22 = self.k2tail(indv, j)
                    if len(ls22) != 0 and len(ls11) != 0:
                        if len(ls22) <= len(ls11):
                            check = ls22.issubset(ls11)
                        else:
                            check = ls11.issubset(ls22)
                        if check:
                            indv = self.mergeRandSub(indv, i, j)
                            #print("kk")
                            break
            if check:
                break
        return indv

    def k2tail(self, indv, nd):
        enum = set()
        if nd in indv.delta:
            for st1 in indv.delta[nd]:
                enum.add(st1)      #For up to k
                if indv.delta[nd][st1] in indv.delta:
                    for st2 in indv.delta[indv.delta[nd][st1]]:
                        enum.add(st1+st2)
        return enum


    def lsenum(self, indv, nd):
        # Basic version. Can be made it recursive
        enum = list()
        if nd in indv.delta:
            for st1 in indv.delta[nd]:
                n1 = indv.delta[nd][st1]
                enum.append(st1)
                if n1 in indv.delta:
                    for st2 in indv.delta[n1]:
                        n2 = indv.delta[n1][st2]
                        enum.append(st1+st2)
                        if n2 in indv.delta:
                            for st3 in indv.delta[n2]:
                                n3 = indv.delta[n2][st3]
                                enum.append(st1 + st2 + st3)
                                if n3 in indv.delta:
                                    for st4 in indv.delta[n3]:
                                        enum.append(st1+st2+st3+st4)
        return enum

# END

# Used just for the evaluation of the result, not in the GA process


    def UnObs2(self, indv):
        self.cntUnObs1 = 0
        nd = indv.Initial
        if nd in indv.delta:
            for st1 in indv.delta[nd]:
                n1 = indv.delta[nd][st1]
                self.chk(st1)
                if n1 in indv.delta:
                    for st2 in indv.delta[n1]:
                        n2 = indv.delta[n1][st2]
                        self.chk(st1 + st2)
                        if n2 in indv.delta:
                            for st3 in indv.delta[n2]:
                                n3 = indv.delta[n2][st3]
                                self.chk(st1 + st2 + st3)
                                if n3 in indv.delta:
                                    for st4 in indv.delta[n3]:
                                        self.chk(st1 + st2 + st3 + st4)

    def chk(self, st1):
        obs = st1.rstrip(" ").split(" ")
        for j in range(len(self.step_list)):
            flag = False
            if len(self.step_list[j]) < len(obs):
                flag = True
            else:
                for t in range(len(obs)):
                    if self.step_list[j][t] != obs[t]:
                        flag = True
                        break
            if not flag:
                break
        if flag:
            self.cntUnObs1 += 1
        if not self.cntUnObs1 % 100000:
            print(str(self.cntUnObs1))


    def traces(self, indv):
        nd = indv.Initial
        enum = 0
        if nd in indv.delta:
            for st1 in indv.delta[nd]:
                n1 = indv.delta[nd][st1]
                if n1 in indv.delta:
                    for st2 in indv.delta[n1]:
                        n2 = indv.delta[n1][st2]
                        if n2 in indv.delta:
                            for st3 in indv.delta[n2]:
                                n3 = indv.delta[n2][st3]
                                if n3 in indv.delta:
                                    for st4 in indv.delta[n3]:
                                        enum += 1
        return enum

    def tracesK(self, indv):
        nd = indv.Initial
        enum = 0
        if nd in indv.delta:
            for st1 in indv.delta[nd]:
                n1 = indv.delta[nd][st1]
                if n1 in indv.delta:
                    for st2 in indv.delta[n1]:
                        n2 = indv.delta[n1][st2]
                        if n2 in indv.delta:
                            for st3 in indv.delta[n2]:
                                n3 = indv.delta[n2][st3]
                                if n3 in indv.delta:
                                    for st4 in indv.delta[n3]:
                                            if indv.finalP(indv.delta[n2][st3]):
                                                enum += 1
        return enum
