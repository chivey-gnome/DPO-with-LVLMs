class MetricParser:
    '''
    Implementers should create specific benchmark parsing rules in subclasses. 
    The output of this should be a pair of lists, full of 1s and 0s indicating
    a positive and a negative. First element in the pair is the predictions,
    second element is the ground truth of the benchmark

    @todo use kwargs to not limit this interface
    '''
    def parse(self, args):
        pass

class calculator:
    def __init__(self, parser: MetricParser):
        self.parser = parser

    def parse(self, result):
        self.parse_results = self.parser.parse(result)

    def calculate_results(self):
        pass

