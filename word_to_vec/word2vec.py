from typing import *

import torch
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from .dataset import W2VTrainDataset
import tensorboardX


class Word2Vec(torch.nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.vocab_size = 0
        self.emb_dimension = 0
        self.center_vector, self.context_vectors = None, None

    def prepare(self, vocab_size: int, emb_dimension: int = 30, ):
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.center_vector = Parameter(xavier_normal_(torch.empty(1, emb_dimension, self.vocab_size)),     # (D, V)
                                       requires_grad=True).float()
        self.context_vectors = Parameter(xavier_normal_(torch.empty(1, self.vocab_size, emb_dimension)),   # (V, D)
                                         requires_grad=True).float()

    def forward(self, x: torch.Tensor):     # x: (V, 1)
        center_vector = self.calc_vector(x)                                 # (1, D, V) * (B, V, 1) -> (B, D, 1)
        inner_product = torch.matmul(self.context_vectors, center_vector)   # (1, V, D) * (B, D, 1) -> (B, V, 1)
        return inner_product

    def calc_vector(self, x: torch.Tensor):
        return torch.matmul(self.center_vector, x)                          # (1, D, V) * (B, V, 1) -> (B, D, 1)


class TrainWord2Vec:
    def __init__(self, device):
        self.device = device
        self.w2v = Word2Vec()
        self.optimizer = None
        self.criterion = None

    def prepare(self, vocab_size: int, emb_dimension: int = 30, lr: float = 0.001):
        self.w2v.prepare(vocab_size=vocab_size, emb_dimension=emb_dimension)
        self.optimizer = Adam(self.w2v.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.w2v.to(self.device)
        self.criterion.to(self.device)

    def train(self, dataset: W2VTrainDataset, epochs: int, batch_size: int = 16, workers: int = 0):
        validation_size = int(len(dataset)*0.01)
        train_dataset, validation_dataset = random_split(dataset, [len(dataset)-validation_size, validation_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        validation_loader = DataLoader(validation_dataset, shuffle=True, num_workers=workers)

        writer = tensorboardX.SummaryWriter()

        print(f'start training {epochs} epoch')
        for epoch in range(epochs):
            loss_value = 0
            cnt = 0
            for x, y in tqdm(train_loader):
                cur_batch_size = x.shape[0]

                # forward
                x = x.to(self.device)
                y = y.to(self.device)
                inner_product = self.w2v(x)                                         # (B, V, 1)

                output = inner_product.view(cur_batch_size, dataset.vocab_size())       # (B, V)
                loss = self.criterion(output, y)
                loss_value += loss.item()
                cnt += cur_batch_size

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            validation_loss_value = 0
            validation_cnt = 0
            self.w2v.eval()
            for x, y in validation_loader:
                cur_batch_size = x.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
                inner_product = self.w2v(x)
                output = inner_product.view(cur_batch_size, dataset.vocab_size())
                validation_loss_value += self.criterion(output, y).item()
                validation_cnt += cur_batch_size
            self.w2v.train()

            writer.add_scalars('word2vec', {
                'train': loss_value / cnt,
                'validation': validation_loss_value / validation_cnt
            }, epoch)

            print(f'epoch {epoch} complete, validation loss: {validation_loss_value / validation_cnt}')

        writer.close()
        self.w2v.cpu().eval()
        torch.save(self.w2v.state_dict(), 'word2vec.pt')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--texts_cnt', type=int, default=150000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--emb_dimension', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    dataset = W2VTrainDataset(args.reviews, word_storage_path="word_storage.dat", texts_cnt=args.texts_cnt)

    using_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('torch device: ' + str(using_device))

    train = TrainWord2Vec(using_device)
    train.prepare(vocab_size=dataset.vocab_size(), emb_dimension=args.emb_dimension, lr=args.lr)
    train.train(dataset, epochs=args.epochs, batch_size=args.batch_size, workers=args.num_workers)


