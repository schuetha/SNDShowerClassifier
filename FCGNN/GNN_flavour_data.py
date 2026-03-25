import ROOT
from tqdm.auto import tqdm

from NC_CC import NC_CC
from signal_identification import signal_background

import numpy as np
import SndlhcGeo
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import argparse

class GNN_signal_background:
    def __init__(self, input_path_s, output_dir, geo, ranges_s, mode=''):
        self.input_path_s = input_path_s
        self.output_dir = output_dir
        self.tree = ROOT.TChain("cbmsim")
        self.mode = mode
        # kaons = ['K_5_10', 'K_5_10_highstat', 'K_10_20', 
        #          'K_20_30', 'K_30_40', 'K_40_50', 
        #          'K_50_60', 'K_60_70', 'K_70_80',
        #          'K_80_90', 'K_90_100'] # Path directory of Kaons

        # neutrons = ['neu_5_10', 'neu_5_10_highstat', 'neu_10_20',
        #             'neu_20_30', 'neu_30_40', 'neu_40_50',
        #             'neu_50_60', 'neu_60_70', 'neu_70_80',
        #             'neu_80_90', 'neu_90_100'] # Path directory of Neutrons
        
        # for i in tqdm(range(500), desc="Adding files to TChain"):
        #     self.tree.Add((f"{self.input_path}/{i}/sndLHC.Genie-TGeant4_digCPP.root"))
        # kaons_path = f'{self.path_start}{k}/{self.kaons[row]}{self.path_end}{column}/{self.files_name_kaons}'
        # neutrons_path = f'{self.path_start}{n}/{self.neutrons[row]}{self.path_end}{column}/{self.files_name_neutrons}'

        for i in tqdm(range(ranges_s[0], ranges_s[1]), desc="Adding files to TChain: Signal", leave=False):
            if i == 40: continue
            self.tree.Add((f"{self.input_path_s}/{i}/sndLHC.Genie-TGeant4_20240126_digCPP.root"))

        # for kaons, neutrons in tqdm(zip(kaons, neutrons), leave=False, desc="Adding files to TChain: Background"):
        #     for i in tqdm(range(ranges_b[0], ranges_b[1]), leave=False, desc="Adding files to TChain"):
        #         self.tree.Add((f"{self.input_path_b}/kaons/{kaons}/Ntuples/{i}/sndLHC.PG_130-TGeant4_digCPP.root"))
        #         self.tree.Add((f"{self.input_path_b}/neutrons/{neutrons}/Ntuples/{i}/sndLHC.PG_2112-TGeant4_digCPP.root"))
        
        self.geo = SndlhcGeo.GeoInterface(geo)
        self.Scifi = self.geo.modules['Scifi']
        self.Mufi = self.geo.modules['MuFilter']

        # self.geo_s = SndlhcGeo.GeoInterface(geo_s)
        # self.Scifi_s = self.geo_s.modules['Scifi']
        # self.Mufi_s = self.geo_s.modules['MuFilter']

        # self.geo_b = SndlhcGeo.GeoInterface(geo_b)
        # self.Scifi_b = self.geo_b.modules['Scifi']
        # self.Mufi_b = self.geo_b.modules['MuFilter']

        self.n_sys = {1: 'veto', 2: 'us', 3: 'ds', 4: 'scifi'}
        # self.classifier = NC_CC(self.tree)
        self.cols = ['XY', 'Z', 'Energy', 'dettype', 'vertical']
    
    def get_mufi_time_avg(self, hit):
        """Average of left-side and right-side mean times.
        This cancels the light travel time along the bar,
        giving a true interaction time."""
        nSiPMs = hit.GetnSiPMs()
        nSides = hit.GetnSides()
        side_times = []
        for side in range(nSides):
            times = []
            for sipm in range(nSiPMs):
                ch = side * nSiPMs + sipm
                s = hit.GetSignal(ch)
                t = hit.GetTime(ch)
                if s > 0 and t > 0:
                    times.append(t)
            if times:
                side_times.append(sum(times) / len(times))
        if side_times:
            return sum(side_times) / len(side_times)
        return -1.0
    def hits_counts(self, tree):
        count = 0
        for hits in tree.Digi_ScifiHits:
            if not hits.isValid():
                continue
            count += 1

        for hits in tree.Digi_MuFilterHits:
            if not hits.isValid():
                continue

            system = hits.GetSystem()
            if system == 1: 
                continue
            count += 1

        return count

    def processing(self, tree):
        flavour   = []
        features  = []
        # n_hits = { 'XY': [], 'Z': [], 'Energy': [], 'dettype': [], 'verticle': [] }
        plane     = [ 'XY', 'Z', 'Energy', 'dettype', 'verticle', 'time' ]
        # n_hits_y = { 'flavour': [],  { 'Y': [], 'Z': [], 'Energy': [], 'dettype': [] } }
        count     = 0
        all_count = 0
        for i in tqdm(range(tree.GetEntries()), desc="Processing events"):
            try:
                n_hits = { 'XY': [], 'Z': [], 'Energy': [], 'dettype': [], 'verticle': [], 'time': [] }
                tree.GetEntry(i)
                A, B   = ROOT.TVector3(), ROOT.TVector3()
                NC_CCs = NC_CC(tree).classify()
                if NC_CCs == 16: continue

                # if NC_CCs == 23:
                #     all_count += 1
                #     if self.hits_counts(tree) < 300: 
                #         count += 1
                #         continue

                flavour.append(NC_CCs)
                # s_b = signal_background(tree).classify()
                # if s_b == 1:
                #     flavour.append(NC_CCs)
                # elif s_b == 0:
                #     flavour.append(0)
                
                for hits in tree.Digi_ScifiHits:
                    if not hits.isValid():
                        continue

                    station = hits.GetStation()
                    detID   = hits.GetDetectorID()
                    vert    = hits.isVertical()
                    energy  = hits.GetEnergy()
                    self.Scifi.GetSiPMPosition(detID, A, B) # A is left(top), B is right(bottom)
                    t       = hits.GetTime(0)

                    if vert:
                        n_hits['XY'].append((A.X() + B.X()) / 2)
                        n_hits['Z'].append((A.Z() + B.Z()) / 2)
                        n_hits['Energy'].append(energy)
                        n_hits['dettype'].append(4)
                        n_hits['verticle'].append(vert)
                        n_hits['time'].append(t)

                    else:
                        n_hits['XY'].append((A.Y() + B.Y()) / 2)
                        n_hits['Z'].append((A.Z() + B.Z()) / 2)
                        n_hits['Energy'].append(energy)
                        n_hits['dettype'].append(4)
                        n_hits['verticle'].append(vert)
                        n_hits['time'].append(t)
                    
                for hits in tree.Digi_MuFilterHits:
                    if not hits.isValid():
                        continue

                    system = hits.GetSystem()
                    if system == 1: 
                        continue

                    detID  = hits.GetDetectorID()
                    vert   = hits.isVertical()
                    energy = hits.GetEnergy()
                    self.Mufi.GetPosition(detID, A, B)
                    
                    if vert:
                        n_hits['XY'].append((A.X() + B.X()) / 2)
                        n_hits['Z'].append((A.Z() + B.Z()) / 2)
                        n_hits['Energy'].append(energy)
                        n_hits['dettype'].append(system)
                        n_hits['verticle'].append(vert)
                        n_hits['time'].append(self.get_mufi_time_avg(hits))

                    else:
                        n_hits['XY'].append((A.Y() + B.Y()) / 2)
                        n_hits['Z'].append((A.Z() + B.Z()) / 2)
                        n_hits['Energy'].append(energy)
                        n_hits['dettype'].append(system)
                        n_hits['verticle'].append(vert)
                        n_hits['time'].append(self.get_mufi_time_avg(hits))

                stacked = np.column_stack([ n_hits[s] for s in plane ]).astype(np.float64)   # (N, row, 5)
                features.append(torch.from_numpy(stacked))
            except Exception as e: print(f"\nCrash in event {i}, skipped\n {e}")
        y = torch.as_tensor(flavour, dtype=torch.int8)
        # print(f"NC < 300: {count}/{all_count}")
        return features, y

    def save_pt(self, filename=f"GNN_data.pt"):
        X_list, y = self.processing(self.tree)
        if self.mode != '':
            filename = f"GNN_data_{self.mode}_small_correct_all_flavour.pt"
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, filename)
        payload = {
            "features": X_list,                  # list[FloatTensor (n_hits_i, 5)]
            "flavours": y,                       # Int16 tensor (N,)
            "feature_name": self.cols            # list of column names
        }
        torch.save(payload, out_path)
        print(f"Saved PyTorch file: {out_path}")
        return out_path

# python3 FCGNN/GNN_flavour_data.py -m train
# python3 FCGNN/GNN_s_b.py -m val
# python3 FCGNN/GNN_s_b.py -m predic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converted data into pt')
    parser.add_argument('-m', '--mode', help='Mode of the data set', type=str, default='train')
    args          = parser.parse_args()

    input_path_s  = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000"

    ranges_s      = { 'train': (0, 400), 'val': (251, 350), 'predic': (351, 400) }[args.mode]
    
    outputs       = '/eos/user/s/schuetha/signal_flavour'
    geo           = "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V0_2022.root"

    GNN_signal_background(input_path_s, outputs, geo, ranges_s, args.mode).save_pt()