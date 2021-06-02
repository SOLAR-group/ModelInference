from src.NLP import Jsonrpc
import time
import pickle
import json
import re
import os
import math
import collections
from joblib import Parallel, delayed

'''
    Class with all the method to communicate with the Bugzilla API in order to retrieve the bugs comment
'''


class BugzillaCall:

    def __init__(self, setting):
        self.setting = setting

    # 1Method to retrieve the products available
    def findProd(self):
        print("Get Product\n")
        lim = 500

        # Call to get all the valid ID program, necessary for the next call
        retrieve = Jsonrpc.try_rpc(Jsonrpc.get_query,
                                   maxtry=2,
                                   setting=self.setting,
                                   method="Product.get_selectable_products")

        resp = []

        # Call to get all the valid Product Name (based on the previous Product ID founded)
        if len(retrieve["result"]["ids"])>lim:
            resp = self.prodSub(retrieve["result"]["ids"], lim)
        else:
            query_params = {"ids": retrieve["result"]["ids"]}
            resp = (Jsonrpc.try_rpc(Jsonrpc.get_query,
                                    maxtry=2,
                                    setting=self.setting,
                                    method="Product.get",
                                    **query_params))["result"]["products"]

        tmpdict = {}

        # Extract name from the URL, just to create the nameID for the file
        name = re.search('[\.][a-zA-Z0-9_]*[\.]', self.setting["url_jsonrpc"]).group(0).replace(".", "")
        if name is None:
            name = re.search('[////][a-zA-Z0-9_]*[\.]', self.setting["url_jsonrpc"]).group(0).replace(".", "").replace("/","")

        # NameProduct: ID
        for i in range(len(resp)):
            tmpdict.update({resp[i]["name"]: resp[i]["id"]})

        f1 = open('data/program/Product_'+name+'.txt', 'w')
        for key in sorted(tmpdict):
            f1.write("%s \t\t %s \n" % (key, tmpdict[key]))

        return list(tmpdict.keys())

    def prodSub(self, retr, lim):
        resp = []

        for i in range(0, len(retr), lim):
            if (i + lim) <= len(retr):
                j = i + lim
            else:
                j = len(retr)

            query_params = {"ids": retr[i:j]}
            resp += (Jsonrpc.try_rpc(Jsonrpc.get_query,
                                     maxtry=2,
                                     setting=self.setting,
                                     method="Product.get",
                                     **query_params))["result"]["products"]
        return resp

    def findProductComponent(self, product, flag):

        # Get all the component of a specified program
        print ("Get component\n")

        query_params = {"names": product}
        resp = (Jsonrpc.try_rpc(Jsonrpc.get_query,
                                maxtry=2,
                                setting=self.setting,
                                method="Product.get",
                                **query_params))["result"]["products"]

        lsnameComp = []

        # Find the right program and then retrieve all is component
        for i in range(len(resp)):
            if resp[i]["name"] == product:
                lsnameComp = resp[i]["components"]
                break

        resultcomp=[]
        for j in range(len(lsnameComp)):
            #if lsnameComp[j]["is_active"]:
            resultcomp.append(lsnameComp[j]["name"])

        if flag:
            f2 = open('data/program/Product_' + product + '_comp.txt', 'w')
            resultcomp.sort()
            for k in range(len(resultcomp)):
                f2.write("%s \n" % (resultcomp[k]))

        return resultcomp

    # 2Method to search a list of bug with specific parameters. Sequential version
    def findBugID(self, Product, date_start, limit, compn, resolution, offs, flagJS):
        #
        print("Search Bug")

        start = time.time()

        query_params = {"program": Product, "status": ["VERIFIED", "CLOSED", "RESOLVED"], "component": compn, "resolution": resolution, "limit": limit, "offset": offs,
                        "creation_time": date_start}
        resb = (Jsonrpc.try_rpc(Jsonrpc.get_query,
                                maxtry=1,
                                setting=self.setting,
                                method="Bug.search",  # The search method (UNSTABLE) can change in future version of Bugzilla
                                **query_params))["result"]["bugs"]
        end = time.time()

        # If we have searched all component, we named the file plus, otherwise name it like the component
        if isinstance(compn, list):
            compn = "plus"

        if flagJS:
            json.dump(resb, fp=open('data/bug/BugRawDataJ_'+Product+'_'+compn+'.txt', 'w'), indent=4)

        print('Time {}:{}'.format(*divmod(round(end-start), 60)))
        print("N. Bug: "+str(len(resb))+"\n")

        D = []
        bestStoreB = {}
        # Change the scructure of the dictionary, make it like {IDbug: allInfo}
        for i in range(len(resb)):
            D.append(resb[i]["id"])
            bestStoreB[int(resb[i]["id"])] = resb[i]

        if flagJS:
            json.dump(bestStoreB, fp=open('data/bug/BugRawDataJ_'+Product+'_'+compn+'.txt', 'w'), indent=4)

        return D

    # 2Method to search a list of bug with specific parameters. Parallel version
    def findBugIDParallel(self, Product, date_start, limit, compn, resolution, flagJS, n_jobs, limitall=10000):

        if limit < 500 & limit != 0:
            self.findBugID(Product, date_start,  limit, compn, resolution, 0, 0)
        else:
            step = math.floor(limit/n_jobs)
            if step > 5000:
                step = 5000

            print("Search Bug (parallel)")
            start = time.time()
            allBugls=[]

            if limit:
                bugls = Parallel(n_jobs)(delayed(self.BugIDParSub)(Product, date_start, step, i, limit,  compn, resolution) for i in range(0, limit, step))
                for x in range(len(bugls)):
                    allBugls += bugls[x]
            else:
                NbugIter = 10000
                initIter = 0
                step = round(limitall/n_jobs)
                while True:
                    bugls = Parallel(n_jobs)(delayed(self.BugIDParSub)(Product, date_start, step, i, NbugIter, compn, resolution) for i in range(initIter, NbugIter, step))
                    for x in range(len(bugls)):
                        allBugls += bugls[x]
                    print ("\r"+str(len(allBugls)), end="")
                    if len(bugls[n_jobs-1]) == 0 or (len(allBugls) % limitall) or len(allBugls) >= 240000:
                        print ("\rfinish         ")
                        break
                    NbugIter += limitall
                    initIter += limitall

            end = time.time()

            if isinstance(compn, list):
                compn = "plus"

            bestStore={}

            for i in range(len(allBugls)):
                bestStore[allBugls[i]["id"]] = allBugls[i]

            # The parallel version sometimes can return duplicate bug, in case like this we remake the search with the
            # sequential version for better result
            if len(allBugls) == len(bestStore) or len(bestStore)>=25000:
                print('Time {}:{}'.format(*divmod(round(end - start), 60)))
                print("N. Bug: " + str(len(allBugls)))
                print("Ok\n")

                if flagJS:
                    json.dump(bestStore, fp=open('data/bug/BugRawDataJp_' + Product + '_' + compn + '.txt', 'w'), indent=4)

                return list(bestStore.keys())
            else:
                print("Better do sequential version")
                return self.findBugID(Product, date_start, limit, compn, resolution, flagJS)

    # 2Method: parallel subroutine
    def BugIDParSub(self, Product, date_start, steps, i, stop, compn="", resolution=""):

        if (steps + i) > stop:
            steps = stop - i

        query_params = {"program": Product, "status": ["VERIFIED", "CLOSED", "RESOLVED"], "component": compn, "resolution": resolution, "limit": steps, "offset": i,
                        "creation_time": date_start}
        resb = (Jsonrpc.try_rpc(Jsonrpc.get_query,
                                maxtry=1,
                                setting=self.setting,
                                method="Bug.search",  # The search method (UNSTABLE) can change in the future version of Bugzilla
                                **query_params))["result"]["bugs"]

        return resb

    # 3Method. Parallel version
    def findBugCommentParallel(self, listBugID, Product, compn, flagP, flagJS, n_jobs):
        print("Retrieve Comment (parallel)")

        commentBug={}
        start = time.time()

        commls=Parallel(n_jobs)(delayed(self.BugCommentParSub)(listBugID, i) for i in range(0, len(listBugID), 500))

        for x in range(len(commls)):
            commentBug.update(commls[x])

        end = time.time()
        print('\rTime {}:{}'.format(*divmod(round(end-start), 60)))
        print("N. Bug: "+str(len(commentBug))+"\n")

        # if was choosen more than one component the name file stored finish with plus,
        # instead of a list of all the component
        if isinstance(compn, list):
            compn = "plus"

        # To save space, one can choose the pickle version to store
        if flagP:
            ff = open('data/bug/BugRawCommPp_'+Product+'_'+compn+'.pkl', 'wb')
            pickle.dump(commentBug, ff)
            ff.close()

        if flagJS:
            json.dump(commentBug, fp=open('data/bug/BugRawCommJp_'+Product+'_'+compn+'.txt', 'w'), indent=4)

        return commentBug

    # 3.1Method parallel subroutine
    def BugCommentParSub(self, ids, i):

        if (i + 500) <= len(ids):
            j = i + 500
        else:
            j = len(ids)

        query_params = {"ids": ids[i:j]}
        commentBug = (Jsonrpc.try_rpc(Jsonrpc.get_query,
                                      maxtry=1,
                                      setting=self.setting,
                                      method="Bug.comments",  # The 'comment' method (STABLE) will not change in the future ver
                                      **query_params))["result"]["bugs"]

        print("\r"+ str(i)+"        ", end="")

        for j in commentBug.keys():
            commentBug[j]["comments"][1:len(commentBug[j]["comments"])] = []

        return commentBug

    def checkDir(self):
        filename = '../../data/bug/dataset/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = '../../data/Loop/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = 'data/program/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = '../../data/cluster/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def findStepsSeqPar(self, Product, compn, lsID, commt, flagP, flagS):
        print ("Steps extraction")

        # if was choosen more than one component the name file stored finish with plus
        if isinstance(compn, list):
            compn = "plus"

        # Setting lsID = -1, the comment are loaded directly from the previously created file
        if lsID == -1:
            # okKey = []
            # bugID = json.load(open("data/bug/BugRawDataJp_"+Product+"_"+compn+".txt", "r"))
            # for key in bugID.keys():
            #     if not bugID[key]["resolution"] == "
            start = time.clock()
            commt = json.load(open("data/bug/BugRawCommJp_"+Product+"_"+compn+".txt", "r"))
            lsID = list(commt.keys())
            end = time.clock()
            print("Time load: %.2gs" % (end - start))

        start = time.clock()

        commtok = {}
        commt_token = {}
        if len(lsID) < 5000:
            n_jobs = 1
        else:
            n_jobs = 2

        step = round(len(lsID)/n_jobs)
        resteps = Parallel(n_jobs)(delayed(self.StepsParSub)(lsID, commt, i, step) for i in range(0, len(lsID), step))

        for x in resteps:
            commtok.update(x[0])
            commt_token.update(x[1])

        end=time.clock()
        print("Time extr. %.2gs" % (end - start))
        print("N. Bug: " + str(len(commt_token)))

        if flagP:
            ff = open('data/bug/dataset/BugStepsPp_'+Product+'_'+compn+'.pkl', 'wb')
            pickle.dump(commt_token, ff)
            ff.close()
        if flagS:
            ordBug = collections.OrderedDict()
            bugIDnum = dict()
            for k in sorted(commt_token, key=int):
                ordBug[k] = commt_token[k]
                bugIDnum
            json.dump(ordBug, fp=open('data/bug/dataset/BugStepsJp_'+Product+'_'+compn+'.txt', 'w'), indent=4)
            #json.dump(commtok, fp=open('data/bug/dataset/BugStepsRawJp_'+Product+'_'+compn+'.txt', 'w'), indent=4)

        return commt_token


    def StepsParSub(self, lsID, commt, i, step):

        if (i + step) <= len(lsID):
            j = i + step
        else:
            j = len(lsID)
        commtok1 = {}
        commt_token1 = {}
        for i in lsID[i:j]:
            try:
                # Pattern to search: This are two of the most common template,
                # but probably you have to find the one used in your bug system
                text = re.search('Steps to reproduce:\n(.*)\nCurrent behavior',
                                 commt[str(i)]["comments"][0]["text"], re.DOTALL | re.IGNORECASE)
            except IndexError:
                continue        # In case of comment blank or bad bug
            if text == None:
                 text = re.search('Steps to reproduce:\n(.*)\nActual result',
                                  commt[str(i)]["comments"][0]["text"], re.DOTALL | re.IGNORECASE)
            if not text == None:
                text = text.group(1)
                entries = re.split("\n+", text)
                entok = []
                for n in range(len(entries)):
                    # Check if the step have at least one character (in order to eliminate steps like this "1. ....")
                    if not re.search("[A-Z]", entries[n], re.IGNORECASE) == None:
                        # Check if the step begin with a number
                        # this can eliminate some 'good' bug report with non numbered 'steps to reproduce' (not many),
                        # but can make a better refinement of the numbered one
                        if str.isnumeric(entries[n][0]):
                            entok.append(entries[n])
                if len(entok) > 1:
                    commtok1[i] = text
                    commt_token1[i] = entok
        return (commtok1, commt_token1)


    #
    #Sequential versions
    #

    def findSteps(self, Product, compn, lsID, commt, flagP, flagS):
        print("Steps extraction")

        # if was choosen more then one component the name file stored finish with plus
        if isinstance(compn, list):
            compn = "plus"

        # Setting lsID=-1, the comment are loaded directly from previously created file
        if lsID==-1:
            # okKey = []
            # bugID = json.load(open("data/bug/BugRawDataJp_"+Product+"_"+compn+".txt", "r"))
            # for key in bugID.keys():
            #     if not bugID[key]["resolution"] == "
            start = time.clock()
            commt = json.load(open("data/bug/BugRawCommJp_" + Product + "_" + compn + ".txt", "r"))
            lsID = list(commt.keys())
            end = time.clock()
            print("Time load: %.2gs" % (end - start))

        start = time.clock()

        commtok = {}
        commt_token = {}

        for i in lsID:
            try:
                # Pattern to search: This are two of the most common template,
                # but probably you have to find the one used in your bug system
                text = re.search('Steps to reproduce:\n(.*)\nCurrent behavior',
                                 commt[str(i)]["comments"][0]["text"], re.DOTALL | re.IGNORECASE)
            except IndexError:
                continue        #In case of comment blank or bad bug
            if text == None:
                 text = re.search('Steps to reproduce:\n(.*)\nActual result',
                                  commt[str(i)]["comments"][0]["text"], re.DOTALL | re.IGNORECASE)
            if not text == None:
                text = text.group(1)
                entries = re.split("\n+", text)
                entok = []
                for n in range(len(entries)):
                    # Check if the step have at least one character (in order to eliminate steps like this "1. ....")
                    if not re.search("[A-Z]", entries[n], re.IGNORECASE) == None:
                        # Check if the step begin with a number
                        # this can eliminate some 'good' bug report with non numbered 'steps to reproduce' (not many),
                        # but can make a better refinement of the numbered one
                        if str.isnumeric(entries[n][0]):
                            entok.append(entries[n])
                if len(entok) > 1:
                    commtok[i] = text
                    commt_token[i] = entok

        end=time.clock()
        print("Time %.2gs" % (end - start))
        print("N. Bug: " + str(len(commt_token)))

        if flagP:
            ff = open('data/bug/dataset/BugStepsPp_'+Product+'_'+compn+'.pkl', 'wb')
            pickle.dump(commt_token, ff)
            ff.close()
        if flagS:
            json.dump(commt_token, fp=open('data/bug/dataset/BugStepsJp_'+Product+'_'+compn+'.txt', 'w'), indent=4)
            json.dump(commtok, fp=open('data/bug/dataset/BugStepsRawJp_'+Product+'_'+compn+'.txt', 'w'), indent=4)

        #return commt_token

    # 3Method to retrieve the comment of a list of bugs. Sequential version
    def findBugComment(self, ids, Product, compn, flagP, flagJS):
        print("Retrieve Comment")

        start = time.time()
        commentBug = {}

        # Made multiple calls with a sub-range of the IDs because a call with all together make the URL too long
        for i in range(0, len(ids), 500):
            if (i + 500) <= len(ids):
                j = i + 500
            else:
                j = len(ids)

            query_params = {"ids": ids[i:j]}
            resp = (Jsonrpc.try_rpc(Jsonrpc.get_query,
                                    maxtry=1,
                                    setting=self.setting,
                                    method="Bug.comments",  # The 'comment' method (STABLE) will not change in the future
                                    **query_params))["result"]["bugs"]

        for j in resp.keys():
            resp[j]["comments"][1:len(resp[j]["comments"])] = []
        commentBug.update(resp)

        end = time.time()
        print('Time {}:{}'.format(*divmod(round(end-start), 60)))
        print("N. Bug: \n"+str(len(commentBug)))

        # if was choosed more then one component the name file stored finish with plus
        if isinstance(compn, list):
            compn = "plus"

        # To save space, you can choose the pickle version to store
        if flagP:
            ff = open('data/bug/BugRawCommP_'+Product+'_'+compn+'.pkl', 'wb')
            pickle.dump(commentBug, ff)
            ff.close()

        if flagJS:
             json.dump(commentBug, fp=open('data/bug/BugRawCommJ_'+Product+'_'+compn+'.txt', 'w'), indent=4)

        return commentBug


