from torch.utils.data import DataLoader
from iter_dataset import Dataset
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
np.random.seed(7)

def train():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    dataset = Dataset()
    dataloader = DataLoader(dataset, batch_size=128)
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
        #mention_embedding = tokenizer(mention, return_tensors="pt")
        #sentence_embedding = tokenizer(sentence, return_tensors="pt")
        #gold_embedding = tokenizer(gold, return_tensors="pt")
        #negative_embedding = tokenizer(negative, return_tensors="pt")

        pos = torch.sigmoid(torch.dot(torch.tensor(mention_embedding), torch.tensor(gold_embedding)))
        neg = torch.sigmoid(torch.dot(torch.tensor(mention_embedding), torch.tensor(negative_embedding)))
        pos_labels = torch.ones(pos.size(1)) - 0.2
        neg_labels = torch.ones(neg.size(1)) - 0.7

        loss = criterion(torch.cat([pos, neg], dim=0), torch.cat([pos_labels, neg_labels], dim=0))
        loss.backward()
        optimizer.step()

        print (loss.item())

train()