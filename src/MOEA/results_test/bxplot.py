from src.MOEA.deap.FSMop import *
# from hv import *
from evoalgos import performance
import time
import pickle
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

# agg backend is used to create plot as a .png file
mpl.use('agg')


class TestR:
    def __init__(self, prod):
        self.product = prod

    def set_bp(self, bp):
        # change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set(color='#000000', linewidth=2)
            # change fill color
            box.set(facecolor='#ffffff')

        # change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#FF0000', linewidth=2)

        # change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)

    def compute_res(self):
        referP = [0, 0, 0]
        set_nsga = set()
        set_ga = set()
        set_ga5 = set()
        fl = True
        alg_values = {}
        alg_values["NSGAII"] = [[], [], []]
        alg_values["NSGAIII"] = [[], [], []]
        alg_values["MOEAD"] = [[], [], []]


        if not fl:
            referP = [426, 975, 6000000]
            print(str(referP))
            # hv = HyperVolume(referP)
            hv = performance.FonsecaHyperVolume(referP)
            #front_nsga = [list(item) for item in set_nsga]
            #front_ga = [list(item) for item in set_ga]
            # vol_nsga = hv.compute(front_nsga)
            #vol_klfa = hv.assess_non_dom_front([[0, 88, 6000000]]*70) IO
            # print(vol_klfa) IO

        if fl:
            for alg in ["NSGAII", "NSGAIII", "MOEAD"]:
                for i in range(1, 31):
                    print("\r" + str(i) + "        ", end="")
                    front_file = open('../../../data/' + product + '/'
                                      + alg + '/' + str(i) + '_EAfront' + alg + '_' + product + '.pkl', 'rb')
                    pareto = pickle.load(front_file)
                    front_file.close()
                    for fit_vector in pareto.get("F"):
                        #set_nsga.add(pareto.keys[k].values)
                        for j in range(3):
                            alg_values[alg][j].append(fit_vector[j])
                    del pareto

        data_pl_UnRec = [alg_values["NSGAII"][0], alg_values["NSGAIII"][0], alg_values["MOEAD"][0]]
        data_pl_Size = [alg_values["NSGAII"][1], alg_values["NSGAIII"][1], alg_values["MOEAD"][1], 475]
        data_pl_UnObs = [alg_values["NSGAII"][2], alg_values["NSGAIII"][2], alg_values["MOEAD"][2]]

        font = {'size': 12}

        mpl.rc('font', **font)

        # Create a figure instance
        fig = plt.figure(1, figsize=(4, 4.2))
        # Create an axes instance
        ax = fig.add_subplot(111)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
        ax.set_xlabel(product)
        # Create the boxplot
        bp = ax.boxplot(data_pl_UnRec)
        ## add patch_artist=True option to ax.boxplot()
        ## to get fill color
        bp = ax.boxplot(data_pl_UnRec, patch_artist=True)
        self.set_bp(bp)
        ax.set_xticklabels(['NSGA-II', 'NSGA-III', 'MOEA/D'])
        #ax.set_xlim(0.5, 2 + 0.5)
        # Save the figure
        fig.savefig('../../../data/boxplots/'+ self.product +'UnRec.png', bbox_inches='tight')

        fig = plt.figure(2, figsize=(4, 4.2))
        # Create an axes instance
        ax = fig.add_subplot(111)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
        ax.set_xlabel(product)
        # Create the boxplot
        bp = ax.boxplot(data_pl_Size)
        ## add patch_artist=True option to ax.boxplot()
        ## to get fill color
        bp = ax.boxplot(data_pl_Size, patch_artist=True)
        self.set_bp(bp)
        ax.set_xticklabels(['NSGA-II', 'NSGA-III', 'MOEA/D', 'KLFA'])
        #ax.set_xlim(0.5, 2 + 0.5)
        # Save the figure
        fig.savefig('../../../data/boxplots/'+ self.product +'Size.png', bbox_inches='tight')

        fig = plt.figure(3, figsize=(4, 4.2))
        # Create an axes instance
        ax = fig.add_subplot(111)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
        ax.set_title('2E+07 - KLFA', fontsize = 11)
        ax.set_xlabel(product)
        # Create the boxplot
        bp = ax.boxplot(data_pl_UnObs)
        ## add patch_artist=True option to ax.boxplot()
        ## to get fill color
        bp = ax.boxplot(data_pl_UnObs, patch_artist=True)
        self.set_bp(bp)
        ax.set_xticklabels(['NSGA-II', 'NSGA-III', 'MOEA/D'])
        #ax.set_xlim(0.5, 2 + 0.5)
        # Save the figure
        fig.savefig('../../../data/boxplots/' + self.product +'UnObs.png', bbox_inches='tight')


if __name__ == '__main__':
    software = ["Calendar"]
    startf = time.perf_counter()
    step = list()
    os.makedirs(os.path.dirname("../../../data/boxplots/"), exist_ok=True)
    for product in software:
        print(product + "\n")
        ts = TestR(product)
        ts.compute_res()
        del ts

    print("Done. Time: " + str(datetime.timedelta(seconds=round(time.perf_counter() - startf))))
