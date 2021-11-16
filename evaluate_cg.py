import dataset as D
import sys
from candidate_generation import Candidate_Generator

input_path = sys.argv[1]
conll_path = sys.argv[2]

class Tester:
    def __init__(self):
        self.data = D.CoNLLDataset(input_path, conll_path)
        self.candidates_generator = Candidate_Generator()
        candidates = {}
        for line in self.data.train:
            for meta in self.data.train[line]:
                for candidate in meta['candidates']:
                    candidates[candidate[0]] = self.candidates_generator.model.encode(candidate[0])