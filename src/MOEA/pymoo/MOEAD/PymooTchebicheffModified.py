import autograd.numpy as anp

from pymoo.model.decomposition import Decomposition


class Tchebicheff(Decomposition):

    def _do(self, F, weights, **kwargs):
        if self.nadir_point[2] == 0:  # Avoid division by zero
            self.nadir_point[2] = 1
        v = anp.abs((F - self.ideal_point)/(self.nadir_point - self.ideal_point)) * weights
        tchebi = v.max(axis=1)
        return tchebi
