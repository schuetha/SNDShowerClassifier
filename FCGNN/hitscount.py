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
    def __init__(self, input_path_s, input_path_ss, input_path_b, output_dir, geo, ranges_s, ranges_b, mode=''):
        self.input_path_s = input_path_s
        self.input_path_b = input_path_b
        self.output_dir = output_dir
        self.tree = ROOT.TChain("cbmsim")
        self.mode = mode
        kaons = ['K_5_10', 'K_5_10_highstat', 'K_10_20', 
                 'K_20_30', 'K_30_40', 'K_40_50', 
                 'K_50_60', 'K_60_70', 'K_70_80',
                 'K_80_90', 'K_90_100'] # Path directory of Kaons

        neutrons = ['neu_5_10', 'neu_5_10_highstat', 'neu_10_20',
                    'neu_20_30', 'neu_30_40', 'neu_40_50',
                    'neu_50_60', 'neu_60_70', 'neu_70_80',
                    'neu_80_90', 'neu_90_100'] # Path directory of Neutrons
        
        for i in tqdm(range(ranges_s[0], ranges_s[1]), desc="Adding files to TChain: Signal", leave=False):
            if i == 40: continue
            self.tree.Add((f"{self.input_path_s}/{i}/sndLHC.Genie-TGeant4_20240126_digCPP.root"))

        for kaons, neutrons in tqdm(zip(kaons, neutrons), leave=False, desc="Adding files to TChain: Background"):
            for i in tqdm(range(ranges_b[0], ranges_b[1]), leave=False, desc="Adding files to TChain"):
                self.tree.Add((f"{self.input_path_b}/kaons/{kaons}/Ntuples/{i}/sndLHC.PG_130-TGeant4_digCPP.root"))
                self.tree.Add((f"{self.input_path_b}/neutrons/{neutrons}/Ntuples/{i}/sndLHC.PG_2112-TGeant4_digCPP.root"))
        
        self.geo = SndlhcGeo.GeoInterface(geo)
        self.Scifi = self.geo.modules['Scifi']
        self.Mufi = self.geo.modules['MuFilter']
        self.n_sys = {1: 'veto', 2: 'us', 3: 'ds', 4: 'scifi'}
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
        flavour = { 0: [], 12: [], 14: [], 23: [] }
        features = []
        count = 0
        all_count = 0
        for i in tqdm(range(tree.GetEntries()), desc="Processing events"):
            try :
                tree.GetEntry(i)
                A, B = ROOT.TVector3(), ROOT.TVector3()
                NC_CCs = NC_CC(tree).classify()
                if NC_CCs == 16: continue
                s_b = signal_background(tree).classify()
                if s_b == 1:
                    flavour[NC_CCs].append( self.hits_counts(tree) )
                elif s_b == 0:
                    flavour[0].append( self.hits_counts(tree) )
                
            except Exception as e: print(f"\nCrash in event {i}, skipped\n {e}")
        return flavour

    def save_pt(self, filename=f"GNN_data.pt"):
        class_data = self.processing(self.tree)
        print("Start plotting: ")
        colors = {
                    0:  '#1f77b4',   # blue  - Background
                    12: '#ff7f0e',   # orange - CC nue
                    14: '#2ca02c',   # green  - CC numu
                    23: '#d62728',   # red    - NC
                }

        class_names = {0: '0', 12: '12', 14: '14', 23: '23'}

        fig, ax = plt.subplots(figsize=(14, 10))

        classes = [0, 12, 14, 23]

        # ─── (a) X Distribution ───
        # ax = axes
        for c in classes:
            x_data = class_data[c]
            # if len(x_data) == 0:
            #     continue
            # # Filter out zeros (fake coordinates)
            # # x_data = x_data[x_data != 0]
            # if len(x_data) == 0:
            #     continue
            
            ax.hist(x_data, bins=100, density=True, histtype='step',
                    linewidth=1.2, color=colors.get(c, 'gray'),
                    label=class_names.get(c, str(c)))
        
        ax.set_xlabel('Hits density [number of hits]')
        ax.set_ylabel('Probability')
        ax.set_title('n_hits distribution')
        ax.legend()
        plt.savefig("./FCGNN/plot_hits_count.pdf", dpi=150)
        plt.close()
        print(f"Saved: ./FCGNN/plot_hits_count.pdf")
        print("Finished plot: ")

# python3 hitscount.py -m train
# python3 FCGNN/GNN_s_b.py -m val
# python3 FCGNN/GNN_s_b.py -m predic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converted data into pt')
    parser.add_argument('-m', '--mode', help='Mode of the data set', type=str, default='train')
    args = parser.parse_args()
    
    inputs_path_b = '/eos/experiment/sndlhc/MonteCarlo/NeutralHadrons/FTFP_BERT'
    input_path_s  = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000"
    input_path_ss = "/eos/user/s/schuetha/2022_MC"

    ranges_s = { 'train': (0, 5), 'val': (251, 350), 'predic': (351, 400) }[args.mode]
    ranges_b = { 'train': (1, 5),  'val': (101, 175), 'predic': (176, 400) }[args.mode]
    
    outputs = '/eos/user/s/schuetha/signal_background_data_new_dataset_with_time'
    geo = "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V0_2022.root"

    GNN_signal_background(input_path_s, input_path_ss, inputs_path_b, outputs, geo, ranges_s, ranges_b, args.mode).save_pt()