from .calculator import calculator
from .amber_metrics import AmberMetricParser

'''
This file is incredibly heavily based on AMBER/inference.py.
Most code from that file has been adapted here, to fit into
the framework established within this tool. This code is mostly
not original, except where it had to be moved around to fit into
our architecture.

Please see the LISCENCE file of the AMBER repo at
https://github.com/junyangwang0410/AMBER
for the original author and permissions to redistribute
and modify this code
'''
class AmberCalculator(calculator):
    def __init__(self, args):
        super().__init__(AmberMetricParser(args))

    def calculate_results(self):
        if self.parse_results['chair_num'] != 0:
            CHAIR = round(self.parse_results['chair_score'] / self.parse_results['chair_num'] * 100, 1)
            Cover = round(self.parse_results['safe_cover_score'] / self.parse_results['safe_cover_num'] * 100, 1)
            Ha = round(self.parse_results['hallu_cover_score'] / self.parse_results['hallu_cover_num'] * 100, 1)
            Ha_p = round(100 - self.parse_results['non_hallu_score'] / self.parse_results['non_hallu_num'] * 100, 1)
            print("Generative Task:")
            print("CHAIR:\t\t", CHAIR)
            print("Cover:\t\t", Cover)
            print("Hal:\t\t", Ha_p)
            print("Cog:\t\t", Ha, "\n")
        
        if self.parse_results['qa_num'] != 0:
            precision = float(self.parse_results['tp']) / float(self.parse_results['tp'] + self.parse_results['fp'])
            recall = float(self.parse_results['tp']) / float(self.parse_results['tp'] + self.parse_results['fn'])
            f1 = 2*precision*recall / (precision + recall)
            acc = (self.parse_results['tp'] + self.parse_results['tn']) / (self.parse_results['tp'] + self.parse_results['tn'] + self.parse_results['fp'] + self.parse_results['fn'])
            yes_ratio = (self.parse_results['tp'] + self.parse_results['fp']) / self.parse_results['qa_num']
            print("Descriminative Task:")
            print(f"TP\tFP\tTN\tFN\t\n{self.parse_results['tp']}\t{self.parse_results['fp']}\t{self.parse_results['tn']}\t{self.parse_results['fn']}\nAccuracy: {acc}\nPrecision:\
            {precision}\nRecall: {recall}\nF1 score: {f1}\nYes ratio: {yes_ratio}")