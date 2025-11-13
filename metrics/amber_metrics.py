from .calculator import MetricParser
import json
import nltk
from nltk.stem import WordNetLemmatizer
import os
import json
import math
import spacy
import warnings
import re
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)

class DistriminativeMetrics:
    def __init__(self, name, outfile):
        self.file = outfile
        self.name = name
        self.tp = 0
        self.fn = 0
        self.tn = 0
        self.fp = 0

    def increment(self, truth, answer):
        if truth == 'yes':
            if answer == 'Yes':
                self.tp += 1
            else:
                self.fn += 1
        else:
            if answer == 'No':
                self.tn += 1
            else:
                self.fp += 1

    def print(self):
        print(f"Descriminative Task: {self.name}")
        print(f"TP\tFP\tTN\tFN\t\n{self.tp}\t{self.fp}\t{self.tn}\t{self.fn}\n")
        tot = self.tp + self.fp + self.tn + self.fn
        precision = float(self.tp) / float(self.tp + self.fp)
        recall = (float(self.tp) / float(self.tp + self.fn)) if not math.isclose(self.tp, 0.0) else 0.0
        f1 = 2*precision*recall / (precision + recall) if not math.isclose((self.tp + self.fn), 0.0) else 0.0
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) if not math.isclose((self.tp + self.tn + self.fp + self.fn), 0.0) else 0.0
        yes_ratio = (self.tp + self.fp) / tot
        print(f"Accuracy: {acc}\nPrecision:{precision}\nRecall: {recall}\nF1 score: {f1}\nYes ratio: {yes_ratio}\n\n")
        with open(self.file, 'a') as f:
            f.write(f"Descriminative Task: {self.name}\n")
            f.write(f"TP\tFP\tTN\tFN\t\n{self.tp}\t{self.fp}\t{self.tn}\t{self.fn}\n")
            f.write(f"Accuracy: {acc}\nPrecision:{precision}\nRecall: {recall}\nF1 score: {f1}\nYes ratio: {yes_ratio}\n\n")

class AmberMetrics:
    def __init__(self, outfile):
        self.file = outfile
        self.chair_score = 0
        self.chair_num = 0
        self.safe_cover_score = 0
        self.safe_cover_num = 0
        self.hallu_cover_score = 0
        self.hallu_cover_num = 0
        self.non_hallu_score = 0
        self.non_hallu_num = 0
        self.qa_num = 0
        self.ex = DistriminativeMetrics("Existence", outfile)
        self.pos = DistriminativeMetrics("Positional", outfile)
        self.attr = DistriminativeMetrics("Attribute", outfile)
        self.all = DistriminativeMetrics("All", outfile)
        with open(self.file, 'w') as f:
            f.write(f"beginning of log...\n")

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

class AmberMetricParser(MetricParser):
    '''
    Function copied from source for ease of use - please see the AMBER code
    base for most up to date version.
    '''
    def check_synonyms_word(self, word1, word2):
        token1 = self.nlp(word1)
        token2 = self.nlp(word2)
        similarity = token1.similarity(token2)
        return similarity > self.sim_score

    '''
    Function copied from source for ease of use - please see the AMBER code
    base for most up to date version.
    '''
    def extract_nouns(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
        return nouns

    def parse_response(self, text):
        # inference file contains whole chat - prompt, response and tags.
        # Remove these for purposes of AMBER benchmarking 

        responses = re.split("Assistant: ", text, flags=re.IGNORECASE)
        if len(responses) >= 2:
            return responses[1]
        else:
            return ""

    def __init__(self, args):
        self.nlp = spacy.load("en_core_web_lg")
        association_file = os.path.join(args.amber_path, "relation.json")
        safewords_file = os.path.join(args.amber_path, "safe_words.txt")
        self.annotations_file = os.path.join(args.amber_path, "annotations.json")
        self.sim_score = args.sim_score
        # parse word associations
        self.associations = json.load(open(association_file, 'r', encoding='utf-8'))
        self.hallucination_words = []

        for keyword in self.associations.keys():
            self.hallucination_words.append(keyword)
            for word in self.associations[keyword]:
                self.hallucination_words.append(word)
        
        # parse safe words
        self.global_safe_words = []
        with open(safewords_file, 'r', encoding='utf-8') as safe_file:
            for line in safe_file:
                line = line.split('\n')[0]
                self.global_safe_words.append(line)
                
        # metrics
        self.metrics = AmberMetrics(f"{args.model_name}.txt")


    def parse(self, args):
        inference_data = [json.loads(q) for q in open(args[1], 'r', encoding='utf-8')]
        ground_truth = json.load(open(self.annotations_file, 'r', encoding='utf-8'))

        for i in tqdm(range(len(inference_data))):
            
            id = inference_data[i]['id']
            offset_id = id - 1
            response = self.parse_response(inference_data[i]['response'])
            if ground_truth[offset_id]['type'] == 'generative':
                nouns = self.extract_nouns(response)
                after_process_nouns = []
                for noun in nouns:
                    if noun in self.hallucination_words:
                        after_process_nouns.append(noun)
                
                safe_words = []
                safe_list = []
                for idx, word in enumerate(ground_truth[offset_id]['truth']):
                    safe_words += self.associations[word]
                    safe_list += [idx] * len(self.associations[word])
                    
                ha_words = []
                ha_list = []
                for idx, word in enumerate(ground_truth[offset_id]['hallu']):
                    ha_words += self.associations[word]
                    ha_list += [idx] * len(self.associations[word])
                
                safe_words += ground_truth[offset_id]['truth']
                safe_len = len(ground_truth[offset_id]['truth'])
                safe_list += [0] * safe_len
                safe_flag_list = [0] * len(after_process_nouns)
                
                ha_words += ground_truth[offset_id]['hallu']
                ha_len = len(ground_truth[offset_id]['hallu'])
                ha_list += [0] * ha_len
                
                for idx, noun in enumerate(after_process_nouns):
                    if noun in self.global_safe_words:
                        continue
                    
                    if noun in safe_words:
                        for j in range(len(safe_words)):
                            if noun == safe_words[j]:
                                if j < (len(safe_list) - safe_len):
                                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                                else:
                                    safe_list[j] = 1
                                break
                        continue
                    
                    if noun in ha_words:
                        for j in range(len(ha_words)):
                            if noun == ha_words[j]:
                                if j < (len(ha_list) - ha_len):
                                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                                else:
                                    ha_list[j] = 1
                                break
                    
                    for j, check_word in enumerate(ha_words):
                        if self.check_synonyms_word(noun, check_word):
                            if j < (len(ha_list) - ha_len):
                                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break
                    
                    flag = False
                    for j, check_word in enumerate(safe_words):
                        if self.check_synonyms_word(noun, check_word):
                            flag = True
                            if j < (len(safe_list) - safe_len):
                                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    if flag == True:
                        continue
                
                    safe_flag_list[idx] = 1

                self.metrics.chair_score += sum(safe_flag_list)
                self.metrics.chair_num += len(safe_flag_list)
                self.metrics.safe_cover_score += sum(safe_list[-safe_len:])
                self.metrics.safe_cover_num += len(safe_list[-safe_len:])
                self.metrics.hallu_cover_score += sum(ha_list[-ha_len:])
                self.metrics.hallu_cover_num += len(ha_list[-ha_len:])
                if sum(safe_flag_list) == 0:
                    self.metrics.non_hallu_score += 1
                self.metrics.non_hallu_num += 1
            
            else:
                self.metrics.qa_num += 1
                truth = ground_truth[offset_id]['truth']
                response = response
                self.metrics.all.increment(truth, response)
                if "discriminative-attribute" in ground_truth[offset_id]['type']:
                    self.metrics.attr.increment(truth, response)
                elif "discriminative-hallucination" in ground_truth[offset_id]['type']:
                    self.metrics.ex.increment(truth, response)
                elif "relation" in ground_truth[offset_id]['type']:
                    self.metrics.pos.increment(truth, response)

        return self.metrics