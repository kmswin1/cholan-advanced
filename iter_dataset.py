from torch.utils.data import IterableDataset
import torch

class Dataset(IterableDataset):
    def __init__(self):
        self.candidate2id = torch.load('candidate2id.pt')
        self.id2candidate = torch.load('id2candidate.pt')

    def __len__(self):
        return len(open('training.txt').readlines())

    def __iter__(self):
        text = open('training.txt', 'r')
        for line in text:
            line = line.strip().split('\t')
            yield line[0], line[1], line[2]