from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from gensim import *
from copy import deepcopy
from FAdo.fa import *
import numpy as np
import collections
import math
import pickle
import nltk
import logging
import re
import json
import os


class TextWork:
    def __init__(self, product, compn, threshold, result=None, log=True):
        os.makedirs(os.path.dirname('../../data/cluster/'), exist_ok=True)

        self.k = 0
        self.overlap = math.inf
        self.Product = product
        self.Compn = compn

        self.corpsOrigin = []   # list of all original document (one item = single step)
        self.corpsLoop = []  # list of document (one document = more step/cluster) for the current clustering iteration
        self.clusterAllID = []  # list of ID of the step in each cluster
        self.finalID = []

        self.corpsListStyle = []
        self.numtopic = 0
        self.numtp = 0
        self.n_cl = 0
        self.numfeat = 0
        self.fl_cl = True
        self.stop = True
        self.tt = 3
        self.threshold = np.round(threshold, 5)
        self.map1st = []
        self.step_map = None    # Contain dict of step, after the mapping
        self.tmp_step = None    # Used just until sent_divide
        self.step = collections.OrderedDict()   # Contain dict of the bug, used after sent_divide
        self.unq_w = 0

        if not result:
            self.tmp_step = json.load(fp=open('../../data/bug/dataset/BugStepsJp_' + product + '_' + compn + '.txt', 'r'))
        else:
            self.tmp_step = result

        # Convert key dict from string to int
        bug_int = dict()
        for key in sorted(self.tmp_step, key=int):
            bug_int[int(key)] = self.tmp_step[key]

        self.tmp_step = collections.OrderedDict(sorted(bug_int.items(), key=lambda t: int(t[0])))

        if False:
            for l in range(0, len(self.tmp_step)-2000):
                self.tmp_step.popitem(last=False)

        self.dimStepOrigin = 0
        self.dimBugOrigin = len(self.tmp_step)
        self.numdoc = 0
        self.ovr = 0
        self.time_ovr = 0
        #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.log = log

    def pre_process(self):
        print("Start step preprocessing\n")

        for key in self.tmp_step:
            self.step[key] = self.sent_divide(self.tmp_step[key])
        self.tmp_step = None

        snowstem = SnowballStemmer("english")
        stopword = stopwords.words('english')
        # add other stop word
        stopword += ['tried', 'tri', 'come', 'youll', 'mean', 'like', 'liked', 'else', 'cry', 'say', 'good', 'try',
                     'eg', 'e.g', 'seems', 'seem', 'see', 'bla', 'blub', 'want', 'wanted', 'awesome', 'via', 'around',
                     'sure', 'always']

        for key in sorted(self.step, key=int):
            steptmp = self.step[key]
            to_remove = []
            self.step[key] = []
            self.dimStepOrigin += len(steptmp)
            # single step preprocess
            for i in range(0, len(steptmp)):
                new_step = []
                wt_ap = steptmp[i].replace("'", "")
                #wt_ap = wt_ap.lower().replace("firefox os", "firefoxos")
                tk_w = word_tokenize(wt_ap)
                tokens = [w.lstrip(".- ").rstrip(".1234567890- ") for w in tk_w if len(w.lstrip(".- ").rstrip(".1234567890- ")) > 1]
                for w in tokens:
                    # word that begin with alphab
                    if (w[0].isalpha()) and (w.lower() not in stopword):
                        w = snowstem.stem(w).lower()
                        new_step.append(w)
                if len(new_step):
                    self.step[key].append(new_step)
            # if all step are void, this bug has to be removed

        self.check_null(self.step)

        freq_Word = self.cnt_freq()

        # remove word below a custom freq
        self.rm_once_fq(freq_Word)

        #self.replace_word()

        freq_Word = self.cnt_freq()

        self.trunc_step(freq_Word)

        # append all step of all bug in a single list
        for key in self.step:
            for i in range(0, len(self.step[key])):
                self.corpsOrigin.append(self.step[key][i])

        self.corpsLoop = deepcopy(self.corpsOrigin)

        # give an ID for all step (is simply the index in the original list)
        for i in range(0, len(self.corpsLoop)):
            temp = [i]
            self.clusterAllID.append(temp)

        print("Num Bug origin: "+str(self.dimBugOrigin)+", num Bug after preprocess: "+str(len(self.step)))
        print("Num Step origin: "+str(self.dimStepOrigin)+", num Step after preprocess: "+str(len(self.corpsOrigin)))

        self.numdoc = len(self.clusterAllID)

        json.dump(self.step, fp=open('../../data/bug/dataset/BugStepsJpless_' + self.Product + '_' + self.Compn + '.txt', 'w'), indent=4)

    def cnt_freq(self):
        freq_Word = {}
        for key in self.step:
            for i in range(0, len(self.step[key])):
                for words in self.step[key][i]:
                    if words in freq_Word:
                        freq_Word[words] += 1
                    else:
                        freq_Word[words] = 1
        return freq_Word

    def check_null(self, step):
        kk = list(step.keys())
        for key in kk:
            if not step[key]:
                #print(str(key))
                del step[key]

    def sent_divide(self, steplist):
        newsteplist = list()
        for st in steplist:
            st = st[2:]
            pareth = re.compile("\([^)]*\)", re.DOTALL | re.IGNORECASE) # Delete words in parenthesis
            st = pareth.sub(" ", st)
            for sent in sent_tokenize(st):
                for vr in re.split(",|; | and | - | > | --> | -> | to | or | so ", sent):
                    to = vr.lstrip().rstrip()
                    if len(to) > 1:
                        newsteplist.append(to)
        return newsteplist

    def replace_word(self):
        word_change = {}
        cmmword = {"firefox": ["program", "firefox", "ff", "mozilla", "fennec", "applic", "browser", "app", "night"]}


        # Build a mapping between the synonims and usable word
        for k in cmmword.keys():
            for i in range(0, len(cmmword[k])):
                word_change[cmmword[k][i]] = k

        for key in self.step:
            for i in range(0, len(self.step[key])):
                for j in range(len(self.step[key][i])):
                    if self.step[key][i][j] in word_change:
                        tmp_exc = word_change[self.step[key][i][j]]
                        self.step[key][i][j] = tmp_exc

    def rm_once_fq(self, freq_W):
        cnt_less = 0
        cnt_all = 0
        less = 1    # frequency threshold

        for key in self.step:
            steptmp = self.step[key]
            self.step[key] = []
            for i in range(0, len(steptmp)):
                tmp = []
                for words in steptmp[i]:
                    if freq_W[words] > less:
                        tmp.append(words)
                        cnt_all += 1
                    else:
                        cnt_less += 1
                if len(tmp):
                    self.step[key].append(tmp)

        self.check_null(self.step)

        print("Tot word: " + str(cnt_all + cnt_less) + " (unique:" + str(len(freq_W)) + "), word with freq < " +
              str(less) + ": " + str(cnt_less)+". Remain word: "+str(len(freq_W)-cnt_less))
        self.unq_w = len(freq_W)-cnt_less


    def trunc_step(self, freq_W):
        # decrease lenght of single step to a minimun of 3 word
        n_len = 5
        for key in self.step:
            for i in range(0, len(self.step[key])):
                if len(self.step[key][i]) > n_len:
                    d_f = {}
                    r = None
                    for words in self.step[key][i]:
                        d_f[words] = freq_W[words]
                        r = [(k, d_f[k]) for k in sorted(d_f, key=d_f.get)]
                    if len(d_f) > n_len:
                        for j in range(len(d_f) - n_len):
                            try:
                                self.step[key][i].remove(r[j][0])
                            except IndexError:
                                print("T") #just debug
                            if len(self.step[key][i]) <= n_len:
                                break


    def Corpus_Cluster(self):
        perc = 25
        self.numtopic = round((self.unq_w*perc)/100)  # or to set manually
        if self.numtopic > 400:
            self.numtopic = 400
        #self.numtopic = 100
        print("Num topic: " + str(self.numtopic)+" - Perc: "+ str(perc))
        print("Num thresh: " + str(self.threshold))
        print("\nStart clustering")
        while self.stop:
            self.clusterLoop()
            self.k += 1

    def clusterLoop(self):
        dictionary = corpora.Dictionary(self.corpsLoop)

        modeltf = [dictionary.doc2bow(text) for text in self.corpsLoop]

        tfidf = models.TfidfModel(modeltf)

        corpus_tfidf = tfidf[modeltf]

        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=self.numtopic, power_iters=4, onepass=True)

        corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

        index = similarities.SparseMatrixSimilarity(corpus_lsi, num_features=self.numtopic)

        clustID = []
        clusterW = []
        fl_tech = True
        it_n = 0

        # Insert into the cluster all the step, even if there is already one equal
        if fl_tech:
            t = 0
            for similaritis in index:
                tmpw = []
                num = []
                #sim = np.round(similaritis, 6)
                for i in range(t, len(similaritis)):
                    if similaritis[i] >= self.threshold:
                        for j in range(0, len(self.clusterAllID[i])):
                            if self.clusterAllID[i][j] not in num:
                                num.append(self.clusterAllID[i][j])
                                tmpw += self.corpsOrigin[self.clusterAllID[i][j]]
                if num not in clustID:
                    clusterW.append(tmpw)
                    clustID.append(num)
                t += 1
            self.corpsLoop = clusterW

        # Insert the step into the cluster only if there isn't already one equal
        # else:
        #     for similaritis in index:
        #         tmpw = []
        #         tmpstep = []
        #         num = []
        #         #sim = np.round(similaritis, 6)
        #         t = 0
        #         for i in range(t, len(similaritis)):
        #             iditer = []
        #             if similaritis[i] >= self.threshold:
        #                 for j in range(0, len(self.clusterAllID[i])):
        #                     if self.clusterAllID[i][j] not in num:
        #                         num.append(self.clusterAllID[i][j])
        #                         iditer.append(self.clusterAllID[i][j])
        #                 for t in range(0, len(iditer)):
        #                     if self.corpsOrigin[iditer[t]] not in tmpstep:
        #                         tmpw += self.corpsOrigin[iditer[t]]
        #                         tmpstep.append(self.corpsOrigin[iditer[t]])
        #             t += 1
        #         if num not in clustID:
        #             clusterW.append(tmpw)
        #             clustID.append(num)
        #     self.corpsLoop = clusterW

        # remove the equal cluster, it's util to accelerate the process
        if True:
            to_rm = self.remove_clst_equal(clustID)
            for i in sorted(list(to_rm), reverse=True):
                del clusterW[i]
                del clustID[i]

        print("Len cluster: "+str(len(clustID)))
        # fil = open("data/cluster/tp/" + self.Product + "clusterW" + str(self.k) + ".txt", "w")
        # for i in range(0, len(clustID)):
        #     for j in range(0, len(clustID[i])):
        #         fil.write(str(self.corpsOrigin[clustID[i][j]]))
        #     fil.write("\n")
        # fil.close()

        if clustID == self.clusterAllID or len(clustID) == len(self.clusterAllID):
            set_cl = []
            for j in clustID:
                set_cl.append(set(j))
            stop_ovr = self.ovr_test(set_cl)
            while stop_ovr:
                self.ovr_reduction(set_cl)
                stop_ovr = self.ovr_test(set_cl)

            ln_cl = list()
            for i in range(len(set_cl)):
                ln_cl.append(len(set_cl[i]))
            ln_cl.sort(reverse=True)
            print("Len cl top 10 (bf ss): " + str(ln_cl[0:20]))

            min_cl = 1
            for cl in set_cl:
                if len(cl) <= min_cl:
                    self.finalID.append(list(cl))
            cnt_sg = len(self.finalID)
            for cl in set_cl:
                if len(cl) > min_cl:
                    self.finalID.append(sorted(list(cl)))

            self.single_step(cnt_sg)

            self.formatCorpus()
            self.writeClust()
            self.finalTest()
            self.corpsLoop = []
            self.clusterAllID = []
            self.stop = False
        else:
            self.clusterAllID = clustID

    def remove_clst_equal(self, clustID):
        sett = []
        ID = []
        toremove = set()
        for j in clustID:
            sett.append(set(j))
        #sett_cp = deepcopy(sett)
        for s in range(0, len(clustID)):
            if s not in toremove:
                for t in range(s + 1, len(clustID)):
                    if t not in toremove:
                        if t > s:
                            res = sett[s].intersection(sett[t])
                            if res:
                                if len(res) == len(sett[s]):
                                    toremove.add(s)
                                elif len(res) == len(sett[t]):
                                    toremove.add(t)
        print("Num cluster: "+str(len(clustID))+". Remove:"+str(len(toremove)))
        return toremove

    def ovr_test(self, sett):
        print("\nNum clust: "+str(len(sett)))
        ID = []
        cnt1 = 0
        cnt2 = 0
        ss = set()
        for s in range(0, len(sett)):
            for t in range(s + 1, len(sett)):
                if t > s:
                    res = sett[s].intersection(sett[t])
                    if res:
                        ss.add(s)
                        ss.add(t)
                        if t not in ID:
                            ID.append(t)
                        if s not in ID:
                            ID.append(s)
                        if self.log:
                            cnt2 += 1
                            # print(
                            #     str(s) + " - " + str(t) + ", len id overlap: " + str(len(res)) + "(" + "{0:.2f}".format(
                            #         (len(res) * 100) / len(sett[s])) + "% - " +
                            #     "{0:.2f}".format((len(res) * 100) / len(sett[t])) + "%)")
                            s_pr = (len(res) * 100) / len(sett[s])
                            t_pr = (len(res) * 100) / len(sett[t])
                            if not 20 < s_pr < 80 and not 20 < t_pr < 80:
                                cnt1 += 1

        print("Cluster overlap: " + str(len(ss)) + ", clust pr: "+str(cnt1)+" on "+str(cnt2))
        cnt = 0
        for t in sett:
            if len(t) > 1:
                cnt += 1
        print("Cluster with size > 1 : " + str(cnt)+" "+"{0:.2f}".format((cnt * 100) / len(sett))+"%"+"\n")
        return len(ss)

    def ovr_reduction(self, set_cl):
        to_add = []
        to_remove = []
        for s in range(0, len(set_cl)):
            for t in range(s + 1, len(set_cl)):
                res = set_cl[s].intersection(set_cl[t])
                if res:
                    s_pr = (len(res) * 100) / len(set_cl[s])
                    t_pr = (len(res) * 100) / len(set_cl[t])
                    if s_pr >= 80 and t_pr >= 80:
                        to_add.append(set_cl[s].union(set_cl[t]))
                        to_remove.append(s)
                        to_remove.append(t)
                        set_cl[s] = set_cl[t] = set()
                    elif 20 < s_pr < 80 and 20 < t_pr < 80:
                        set_cl[s].difference_update(res)
                        set_cl[t].difference_update(res)
                        to_add.append(res)
                    else:
                        if s_pr <= t_pr:
                            set_cl[s].difference_update(res)
                        elif s_pr > t_pr:
                            set_cl[t].difference_update(res)

        to_remove.sort(reverse=True)
        for i in to_remove:
            del set_cl[i]

        set_cl += to_add

    def single_step(self, cnt_sg):
        single_clusterLoop = []
        for i in range(0, len(self.finalID)):
            tmp_w = []
            for j in range(0, len(self.finalID[i])):
                tmp_w += self.corpsOrigin[self.finalID[i][j]]
            single_clusterLoop.append(tmp_w)

        dictionary = corpora.Dictionary(single_clusterLoop)

        modeltf = [dictionary.doc2bow(text) for text in single_clusterLoop]

        tfidf = models.TfidfModel(modeltf)

        corpus_tfidf = tfidf[modeltf]

        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=self.numtopic, power_iters=4, onepass=True)

        corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

        index = similarities.SparseMatrixSimilarity(corpus_lsi, num_features=self.numtopic)

        cnt = 0
        for similaritis in index:
            if cnt >= cnt_sg:
                break
            sim = np.round(similaritis, 6)
            sim[0:cnt_sg] = -np.inf
            index = np.argmax(sim)
            self.finalID[index] += self.finalID[cnt]
            cnt += 1

        self.finalID = self.finalID[cnt_sg:]


    def finalTest(self):
        sett = []
        ID = []
        numstep = 0
        for j in self.finalID:
            tmp = set(j)
            sett.append(tmp)
            numstep += len(tmp)
        for s in range(0, len(self.finalID)):
            for t in range(s, len(self.finalID)):
                if t > s:
                    res = sett[s].intersection(sett[t])
                    if res:
                        if t not in ID:
                            ID.append(t)
                        if s not in ID:
                            ID.append(s)
        print("\nNum Cluster: "+str(len(self.finalID)))
        cnt = 0
        for t in sett:
            if len(t) > 1:
                cnt += 1
        print("Cluster with size > 1 : " + str(cnt)+" "+"{0:.2f}".format((cnt * 100) / len(sett))+"%"+"\n")
        if not ID:
            print("OK no overlap")
        if numstep == len(self.corpsOrigin):
            print("OK number of step")
        else:
            print("Error number of step: " + str(numstep) + ", orgin:" + str(len(self.corpsOrigin)))

    def formatCorpus(self):
        for tm in self.finalID:
            if not tm:
                self.finalID.remove(tm)
        for i in range(0, len(self.finalID)):
            tmp = []
            for ind in self.finalID[i]:
                tmp.append(self.corpsOrigin[ind])
            self.corpsListStyle.append(tmp)

    def writeClust(self):
        fil = open("../../data/cluster/" + self.Product + "clusterID.txt", "w")
        for i in range(0, len(self.finalID)):
            fil.write(str(sorted(self.finalID[i], key=int)) + "\n")
        fil.close()
        fil = open("../../data/cluster/" + self.Product + "clusterW.txt", "w")
        for i in range(0, len(self.finalID)):
            for j in range(0, len(self.finalID[i])):
                fil.write(str(self.corpsOrigin[self.finalID[i][j]]))
            fil.write("\n")
        fil.close()

    def mapping1stPhase(self, flagFile=0):

        print("\nStart 1st mapping phase")
        ln_cl = list()
        for i in range(len(self.finalID)):
            ln_cl.append(len(self.finalID[i]))
        ln_cl.sort(reverse=True)
        print("Len cl top 10 (1st): " + str(ln_cl[0:20]))
        listWord = []
        mapp = []
        listStyle = self.corpsListStyle
        file = open("../../data/cluster/" + self.Product + "Mapping1st.txt", "w")
        nume = 0
        for line in listStyle:
            if not line:
                listStyle.remove(line)
            cc = []
            for w in line:
                word = w.copy()
                word += " "
                cc += word
            listWord.append(cc)

        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        bigram_measures = nltk.collocations.BigramAssocMeasures()

        # TODO: reformulate the code
        for i in range(0, len(listStyle)):
            flag = True
            cnf = listStyle[i][0]
            if len(listStyle[i]) > 1:
                for j in range(1, len(listStyle[i])):
                    if cnf != listStyle[i][j]:
                        flag = False
                        break
            if flag:
                mapp.append(listStyle[i][0])
            else:
                td = {}
                maxn = 0  # Freq value
                sizemax = 0  # Dimension step most frequent in the cluster
                sizeSearch = 0
                for j in range(0, len(listStyle[i])):
                    try:
                        td[len(listStyle[i][j])] += 1
                    except KeyError:
                        td[len(listStyle[i][j])] = 1

                    if td[len(listStyle[i][j])] > maxn:
                        maxn = td[len(listStyle[i][j])]
                        sizemax = len(listStyle[i][j])
                        sizeSearch = sizemax
                    elif td[len(listStyle[i][j])] == maxn and len(listStyle[i][j]) < sizemax:
                        sizeSearch = sizemax
                        maxn = td[len(listStyle[i][j])]
                        sizemax = len(listStyle[i][j])
                if sizemax == 1:
                    fdist1 = nltk.FreqDist(listWord[i])
                    res = fdist1.most_common(3)
                    tmp = []
                    if res[0][0] != ' ':
                        tmp.append(res[0][0])
                        mapp.append(tmp)
                    else:
                        tmp.append(res[1][0])
                        mapp.append(tmp)
                elif sizemax >= 2:
                    res = []
                    if sizemax == 2:
                        finder = nltk.BigramCollocationFinder.from_words(listWord[i], window_size=sizeSearch)
                        finder.apply_freq_filter(1)
                        res = finder.nbest(bigram_measures.raw_freq, 4)
                    elif sizemax == 3:
                        finder = nltk.TrigramCollocationFinder.from_words(listWord[i], window_size=sizeSearch)
                        finder.apply_freq_filter(1)
                        res = finder.nbest(trigram_measures.raw_freq, 4)
                    else:
                        if sizeSearch > 5:
                            sizeSearch -= 1
                        finder = nltk.QuadgramCollocationFinder.from_words(listWord[i], window_size=sizeSearch)
                        finder.apply_freq_filter(1)
                        res = finder.nbest(trigram_measures.raw_freq, 4)
                    if res:
                        flag = True
                        flag2 = True
                        backup = None
                        for tup in res:
                            flag = True
                            flag2 = True
                            tmp = []
                            for t in range(0, len(tup)):
                                tmp += tup[t]
                                if tup[t] == " ":
                                    flag = False
                                    if not (t == 0 or t == (len(tup) - 1)):
                                        flag2 = False
                            if flag:
                                mapp.append(list(tup))
                                break
                            if flag2 and not backup:
                                backup = tup
                        if not flag:
                            if backup:
                                tmp0 = [w for w in backup if w != " "]
                                mapp.append(tmp0)
                            else:
                                for w in listStyle[i]:
                                    if len(w) == sizemax:
                                        # tmp.append(w)
                                        mapp.append(w)
                                        break
                    else:
                        # rr = random.randint(0, len(listStyle)-1)
                        tmp = []
                        for w in listStyle[i]:
                            if len(w) == sizemax:
                                # tmp.append(w)
                                mapp.append(w)
                                break
            file.write(str(mapp[len(mapp) - 1]) + "\t:\t" + str(listStyle[i]) + "\n")
            nume += 1
            if nume != len(mapp):
                pass
                print(str(nume))

        self.map1st = mapp
        print(str(nume) + " n :" + str(len(self.map1st)) + "  :" + str(len(mapp)))


    def mapping2stPhase(self):
        print("\nStart 2st mapping phase")
        prod = self.Product.lower()
        comp = self.Compn.lower()
        map_1st = deepcopy(self.map1st)

        word_change = {}
        cmmword = {prod: [ prod ], "press": ["click", "press", "hit", "klick", "hold"],
                   "open": ["fire", "open", "start", "launch", "run_id", "use", "create", "re-start", "restart", "reopen", "re-open"], "type": ["type", "write", "enter"],
                   "document": ["document", "file"], "view": ["show", "view", "see", "observ", "watch", "look"],
                   "text": ["word", "text", "someth"],
                   "termin": ["termin", "cmd"], "close": ["close", "exit", "end", "quit", "kill"],
                   "go": ["go", "goto"], "select": ["select", "choos"], "delet": ["delet", "remov"],
                   "modifi": ["modifi", "chang"], "creat": ["creat", "make"], "move": ["posit", "move", "switch"]}
        #"reopen": ["re-start", "restart", "reopen", "re-open"],

        # Build a mapping between the synonims and usable word
        for k in cmmword.keys():
            for i in range(0, len(cmmword[k])):
                word_change[cmmword[k][i]] = k
        #print("Product name synonim: "+ str(cmmword[prod]))
        #cmmword.clear()

        for i in range(0, len(map_1st)):
            for j in range(0, len(map_1st[i])):
                if map_1st[i][j] in word_change:
                    tmp = word_change[map_1st[i][j]]
                    map_1st[i][j] = tmp

        dictmap = {}
        tmpdict = {}

        for m in range(len(map_1st)):
            strw = "_".join(map_1st[m])
            map_1st[m] = strw
            try:
                dictmap[strw].append(m)
            except KeyError:
                dictmap[strw] = []
                dictmap[strw].append(m)

        # json.dump(dictmap, fp=open("data/cluster/"+self.Product+"Mapping2st.txt","w"))
        fp = open("../../data/cluster/" + self.Product + "Mapping2st.txt", "w")
        cnt = 0
        for k in dictmap.keys():
            if len(dictmap[k]) > 1:
                fp.write(str(k) + " : " + str([self.map1st[dictmap[k][h]] for h in range(len(dictmap[k]))]) + "\n")
                cnt += 1
        fp.close()
        print("mapp2 n>1: " + str(cnt))

        final_map = [None] * len(self.corpsOrigin)

        for i in range(0, len(map_1st)):
            for ind in self.finalID[i]:
                final_map[ind] = map_1st[i]

        print("Num cluster final: " + str(len(set(map_1st))))

        step_map = deepcopy(self.step)
        j = 0
        for key in self.step:
            for p in range(0, len(self.step[key])):
                step_map[key][p] = [final_map[j]]
                j += 1

        self.step_map = collections.OrderedDict()

        for k in step_map:
            stepL = []
            for t in range(0, len(step_map[k])):
                stepL.append(step_map[k][t][0])
            self.step_map[k] = stepL

        ln_cl = list()
        for i in range(len(self.finalID)):
            ln_cl.append(len(self.finalID[i]))
        ln_cl.sort(reverse=True)

        print("Len cl top 10 (2st): "+str(ln_cl[0:20]))
        json.dump(self.step_map, fp=open('../../data/bug/dataset/BugStepsMappedJp_' + self.Product + '_' + self.Compn + '.json', 'w'), indent=4)
        ff = open('../../data/bug/dataset/BugStepsMappedPp_' + self.Product + '_' + self.Compn + '.pkl', 'wb')
        pickle.dump(self.step_map, ff)
        ff.close()

        return self.step_map