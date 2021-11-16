from torch.utils.data import DataLoader
from iter_dataset import Dataset
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
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

        mention_embedding = tokenizer(mention, return_tensors="pt")
        sentence_embedding = tokenizer(sentence, return_tensors="pt")
        gold_embedding = tokenizer(gold, return_tensors="pt")
        negative_embedding = tokenizer(negative, return_tensors="pt")

        pos = torch.sigmoid(torch.dot(mention_embedding, gold_embedding))
        neg = torch.sigmoid(torch.dot(mention_embedding, negative_embedding))
        pos_labels = torch.ones(pos.size(1)) - 0.2
        neg_labels = torch.ones(neg.size(1)) - 0.7

        loss = criterion(torch.cat([pos, neg], dim=-1), torch.cat([pos_labels, neg_labels], dim=-1))
        optimizer.step()

        print (loss.item())

train()