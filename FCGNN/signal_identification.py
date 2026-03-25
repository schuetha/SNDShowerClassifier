import ROOT
from tqdm.auto import tqdm
import numpy as np

class signal_background:
    def __init__(self, tree):
        self.tree = tree
        # print(self.tree, 'Tree in NC_CC')

    # def classify(self):
    def classify(self):
        # if not hasattr(tree, "MCTrack") or len(tree.MCTrack) == 0:
        #     return -1  # or whatever "unknown" label you want
        pdg_code_1 = self.tree.MCTrack[0].GetPdgCode()
        pdg_code_2 = []
        for i in range(5): pdg_code_2.append(self.tree.MCTrack[i].GetPdgCode())

        pdg = {
            12: 'nu_e',
            -12: 'anti_nu_e',
            14: 'nu_mu',
            -14: 'anti_nu_mu',
            16: 'nu_tau',
            -16: 'anti_nu_tau',
            11: 'e-',
            -11: 'e+',
            13: 'mu-',
            -13: 'mu+',
            15: 'tau-',
            -15: 'tau+',
            2212: 'p',
            -2212: 'anti_p',
            2112: 'n',
            -2112: 'anti_n',
            211: 'pi+',
            -211: 'pi-',
            111: 'pi0',
            321: 'K+',
            -321: 'K-',
            130: 'K_L0',
            310: 'K_S0',
            22: 'gamma',
        }
        
        if abs(pdg_code_1) == 12 or abs(pdg_code_1) == 14:
            return 1 # Signal
        
        else:
            return 0 # Background
