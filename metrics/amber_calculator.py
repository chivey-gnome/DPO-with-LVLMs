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
        if self.parse_results.chair_num != 0:
            self.parse_results.print()
        
        if self.parse_results.qa_num != 0:
            self.parse_results.all.print()
            self.parse_results.ex.print()
            self.parse_results.pos.print()
            self.parse_results.attr.print()