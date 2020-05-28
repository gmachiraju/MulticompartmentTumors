import numpy as np
import math
import pdb


class GrowObj:

    def __init__(self, maxIter, kv, O2max, sigma, rates):
        self.maxIter = maxIter
        self.kv = kv
        self.O2max = O2max
        self.sigma = sigma
        self.rates = rates

        self.y1B, self.y1D = rates[0], rates[1]
        self.y20B, self.y20D = rates[2], rates[3]
        self.y1, self.y20 = self.y1B - self.y1D, self.y20B - self.y20D
        self.x1, self.x20 = 1, 20

    def load(self, NmatA, NmatD, NAtotal, NDtotal, eta, NH, Cs, n):
        self.NmatA = NmatA
        self.NmatD = NmatD
        self.NAtotal = NAtotal
        self.NDtotal = NDtotal
        self.eta = eta
        self.NH = NH
        self.Cs = Cs
        self.n = n


class MeshObj:

    def __init__(self, maxIter, kvs, O2maxs, sigma, rates):
        self.maxIter = maxIter
        self.kvs = kvs
        self.O2maxs = O2maxs
        self.sigma = sigma
        self.rates = rates

        self.y1B, self.y1D = rates[0], rates[1]
        self.y20B, self.y20D = rates[2], rates[3]
        self.y1, self.y20 = self.y1B - self.y1D, self.y20B - self.y20D
        self.x1, self.x20 = 1, 20

    def load(self, grow_objs, idxValid, aEnd, nVec, paramSet, a_matValid, d_matValid, eta_matValid):
        self.grow_objs = grow_objs
        self.idxValid = idxValid
        self.aEnd = aEnd
        self.nVec = nVec
        self.paramSet = paramSet
        self.a_matValid = a_matValid
        self.d_matValid = d_matValid
        self.eta_matValid = eta_matValid
        
class Marker2Mice:

    def __init__(self, EC, blood_abund, basal, t_half, Pis, Dis, wis, phi, normalshed, mode="run", dt=1):
        
        self.blood_abund = blood_abund
        
        # for simulations
        self.EC = EC
        self.basal = basal
        self.t_half = t_half
        self.phi = phi
        self.normalshed = normalshed # fRN in Hori et al
        
        if mode == "experiment":
            if self.phi == None:
                print("You need to enter a phi!")
        
        elif mode == "run":
            if isinstance(wis, int):
                C_scale = Dis * wis
                R_scale = Pis * wis
            else:
                C_scale = Dis @ wis
                R_scale = Pis @ wis         

            if EC == 1:
                scale_factor = R_scale
            if EC == 0:
                scale_factor = C_scale

            if self.phi == None:
                kE = np.log(2) / self.t_half
                self.phi = float((kE * blood_abund) / scale_factor)


class ShedObj:

    def __init__(self, marker, NmatA, NmatD, n, maxIter):
        self.marker = marker
        self.NmatA = NmatA
        self.NmatD = NmatD
        if n != 1:
            self.NAtotal = np.sum(NmatA, 1)
            self.NDtotal = np.sum(NmatD, 1)
        else:
            self.NAtotal = NmatA
            self.NDtotal = NmatD
            
        self.n = n
        self.maxIter = maxIter

        
    def load(self, qtmat, qt, qp, basal):
        self.qtmat = qtmat
        self.qt = qt
        self.qp = qp
        self.basal = basal
