from .calculator import calculator
from .pope_metrics import PopeMetricParser

class PopeCalculator(calculator):
    def __init__(self):
        super().__init__(PopeMetricParser())

    def calculate_results(self):
        pos = 1
        neg = 0
        answers, labels = self.parse_results
        yes_ratio = answers.count(1) / len(answers)
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for pred, label in zip(answers, labels):
            if pred == pos and label == pos:
                true_pos += 1
            elif pred == pos and label == neg:
                false_pos += 1
            elif pred == neg and label == neg:
                true_neg += 1
            elif pred == neg and label == pos:
                false_neg += 1

        precision = float(true_pos) / float(true_pos + false_pos)
        recall = float(true_pos) / float(true_pos + false_neg)
        f1 = 2*precision*recall / (precision + recall)
        acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        return f"TP\tFP\tTN\tFN\t\n{true_pos}\t{false_pos}\t{true_neg}\t{false_neg}\nAccuracy: {acc}\nPrecision:\
        {precision}\nRecall: {recall}\nF1 score: {f1}\nYes ratio: {yes_ratio}"