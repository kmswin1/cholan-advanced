from torch.utils.data import DataLoader
from dataset2 import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, InputExample, losses
import torch
import numpy as np
np.random.seed(7)

def train():
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    dataset = Dataset
    dataloader = DataLoader(dataset, shuffle=True, batch_size=128)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        mention, sentence, gold = batch_data
        negative = np.random.choice(dataset.id2candidate, 1)[0]

        mention_embedding = model.encode(mention)
        sentence_embedding = model.encode(sentence)
        gold_embedding = model.encode(gold)
        negative_embedding = model.encode(negative)

        pos = torch.sigmoid(torch.dot(mention_embedding, gold_embedding))
        neg = torch.sigmoid(torch.dot(mention_embedding, negative_embedding))
        pos_labels = torch.ones(pos.size(1)) - 0.2
        neg_labels = torch.ones(neg.size(1)) - 0.7

        loss = criterion(torch.cat([pos, neg], dim=-1), torch.cat([pos_labels, neg_labels], dim=-1))
        optimizer.step()

        print (loss.item())

train()