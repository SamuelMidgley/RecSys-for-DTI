import torch
from torch.utils.data import Dataset, DataLoader


class RatingDataset(Dataset):
    def __init__(self, train, label):
        self.feature_ = train
        self.label_ = label

    def __len__(self):
        # return size of dataset
        return len(self.feature_)

    def __getitem__(self, idx):
        return torch.tensor(self.feature_[idx], dtype=torch.long), torch.tensor(self.label_[idx], dtype=torch.float)


class MatrixFactorization(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.item_factors.weight)

        self.user_biases = torch.nn.Embedding(n_users, 1)
        self.item_biases = torch.nn.Embedding(n_items, 1)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
        return pred.squeeze()
