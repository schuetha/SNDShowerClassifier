"""
NC_CC.py - Neutrino interaction classifier for SND@LHC
======================================================
Classifies events from MC truth (MCTrack) into:
    12  = CC nue   (nu_e or anti_nu_e + primary e-/e+)
    14  = CC numu  (nu_mu or anti_nu_mu + primary mu-/mu+)
    16  = CC nutau (nu_tau or anti_nu_tau + primary tau-/tau+)
    23  = NC       (neutrino present but no primary charged lepton)
    -1  = Background (no neutrino as primary particle)

Uses GetMotherId() == 0 to ensure only direct interaction
products are considered, avoiding misclassification from
secondary leptons (e.g. muons from pion decay in NC events).

Based on ShipMCTrack from FairShip/sndsw:
    - Track 0 = primary particle (MotherId == -1)
    - Direct daughters of primary have MotherId == 0

Usage:
    from NC_CC import NC_CC
    label = NC_CC(tree).classify()
"""

import ROOT

NEUTRINO_PDGS = {12, -12, 14, -14, 16, -16}

# For each neutrino PDG, the charged lepton PDG that proves CC
CC_LEPTON = {
     12:  11,   # nu_e       -> e-
    -12: -11,   # anti_nu_e  -> e+
     14:  13,   # nu_mu      -> mu-
    -14: -13,   # anti_nu_mu -> mu+
     16:  15,   # nu_tau     -> tau-
    -16: -15,   # anti_nu_tau-> tau+
}


class NC_CC:
    def __init__(self, tree, count=False):
        self.tree = tree
        self.count = count

    def classify(self):
        """
        Classify one event using MC truth.

        Returns:
            12  for CC nu_e / anti_nu_e
            14  for CC nu_mu / anti_nu_mu
            16  for CC nu_tau / anti_nu_tau
            23  for NC (any flavour)
            -1  for background (no neutrino primary)
        """
        tracks = self.tree.MCTrack
        if len(tracks) == 0:
            return -1

        primary_pdg = tracks[0].GetPdgCode()

        if primary_pdg not in NEUTRINO_PDGS:
            return -1

        target_lepton = CC_LEPTON[primary_pdg]

        # Scan daughters: only accept direct products of
        # the primary vertex (MotherId == 0) to avoid
        # secondary leptons from hadron decays
        found = False
        for i in range(1, len(tracks)):
            trk = tracks[i]
            if trk.GetPdgCode() == target_lepton and trk.GetMotherId() == 0:
                found = True
                break

        if found:
            flavour = abs(primary_pdg)
            if self.count:
                print("CC: primary=" + str(primary_pdg) + " lepton=" + str(target_lepton) + " -> " + str(flavour))
            return flavour
        else:
            if self.count:
                print("NC: primary=" + str(primary_pdg) + " no lepton " + str(target_lepton) + " at vertex")
            return 23