## Candidate Generation ##
import re
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class Candidate_Generator(torch.nn.Module):
    def __init__(self):
        super(Candidate_Generator, self).__init__()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def forward(self, mention, context, entity_name):
        mention_embedding = torch.tensor(self.model.encode(mention))
        sentence_embedding = torch.tensor(self.model.encode(context))
        entity_name_embedding = torch.tensor(self.model.encode((entity_name)))

        return torch.dot(mention_embedding*sentence_embedding, entity_name_embedding)