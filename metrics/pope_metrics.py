from .calculator import MetricParser
import json

class PopeMetricParser(MetricParser):
    def parse(self, args):
        answers = [json.loads(q) for q in open(args[1], 'r')]
        label_list = [json.loads(q)['label'] for q in open(args[0], 'r')]

        return (self.parse_answers(answers), self.parse_labels(label_list))
    
    def parse_labels(self, labels):
        for i in range(len(labels)):
            if labels[i] == 'no':
                labels[i] = 0
            else:
                labels[i] = 1
        return labels

    def parse_answers(self, answers):
        for answer in answers:
            text = answer['answer']

            # Only keep the first sentence
            if text.find('.') != -1:
                text = text.split('.')[0]

            text = text.replace(',', '')
            words = text.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                answer['answer'] = 'no'
            else:
                answer['answer'] = 'yes'

        pred_list = []
        for answer in answers:
            if answer['answer'] == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)
        return pred_list