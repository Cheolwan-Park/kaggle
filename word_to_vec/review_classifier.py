import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split
import tensorboardX
from tqdm import tqdm
from .lstm import LSTM
from .word2vec import Word2Vec
from .dataset import ReviewClassifierDataset


class ReviewClassifier(nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, w2v: Word2Vec):
        super(ReviewClassifier, self).__init__()
        self.max_seq_len = max_seq_len
        self.lstm = LSTM(hidden_size, w2v.emb_dimension)
        self.w2v = w2v

        self.linear1 = nn.Linear(hidden_size, 1024)
        self.linear2 = nn.Linear(1024, 1)

    def forward(self, words: Tensor, device):   # words: (B, max_seq_len, V, 1)
        batch_size = words.shape[0]
        c, h = self.lstm.get_initial_c_and_h(batch_size)
        c, h = c.to(device), h.to(device)

        for i in range(self.max_seq_len):
            x = self.w2v.calc_vector(words[:, i, :])
            _, c, h = self.lstm(x, c, h)

        c = self.linear1(c.squeeze(2))
        c = F.relu(c, inplace=True)
        c = self.linear2(c)
        return torch.sigmoid(c)


class TrainReviewClassifier:
    def __init__(self, device):
        self.device = device
        self.classifier = None
        self.optimizer = None
        self.criterion = None

    def prepare(self, hidden_size: int, max_seq_len: int, w2v: Word2Vec, lr: float = 0.001):
        self.classifier = ReviewClassifier(hidden_size, max_seq_len, w2v)
        self.optimizer = Adam(self.classifier.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = torch.nn.MSELoss()
        self.classifier.to(self.device)
        self.criterion.to(self.device)

    def train(self, dataset: ReviewClassifierDataset, epochs: int, batch_size: int = 16, ckpt_interval: int = 5):
        validation_size = int(len(dataset)*0.01)
        train_dataset, validation_dataset = random_split(dataset, [len(dataset)-validation_size, validation_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

        writer = tensorboardX.SummaryWriter()

        print(f'start training {epochs} epoch')
        for epoch in range(epochs):
            loss_value = 0
            cnt = 0
            for words, sentiment in tqdm(train_loader):
                cur_batch_size = words.shape[0]
                # forward
                words = words.to(self.device)
                y = sentiment.to(self.device)

                predict = self.classifier(words, self.device)

                loss = self.criterion(predict, y)
                loss_value += loss.item()
                cnt += cur_batch_size

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            validation_loss_value = 0
            validation_cnt = 0
            self.classifier.eval()
            for words, sentiment in validation_loader:
                cur_batch_size = words.shape[0]
                words = words.to(self.device)
                y = sentiment.to(self.device)
                predict = self.classifier(words, self.device)
                validation_loss_value += self.criterion(predict, y).item()
                validation_cnt += cur_batch_size
            self.classifier.train()

            writer.add_scalars('review classify', {
                'train': loss_value / cnt,
                'validation': validation_loss_value / validation_cnt
            }, epoch)

            print(f'epoch {epoch} complete, validation loss: {validation_loss_value / validation_cnt}')

            if (epoch + 1) % ckpt_interval == 0:
                torch.save(self.classifier.state_dict(), f'/ckpts/classifier_ckpt_{epoch}.pt')

        writer.close()
        self.classifier.cpu().eval()
        torch.save(self.classifier.state_dict(), 'classifier.pt')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', type=str, required=True)
    parser.add_argument('--word_storage', type=str, required=True)
    parser.add_argument('--w2v', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=64)
    args = parser.parse_args()

    dataset = ReviewClassifierDataset(args.reviews, args.word_storage, args.max_seq_len)

    using_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('torch device: ' + str(using_device))

    train = TrainReviewClassifier(using_device)
    w2v = Word2Vec()
    w2v.prepare(vocab_size=dataset.vocab_size, emb_dimension=128)
    w2v.load_state_dict(torch.load(args.w2v))
    train.prepare(hidden_size=args.hidden_size, max_seq_len=args.max_seq_len, w2v=w2v, lr=args.lr)
    train.train(dataset, epochs=args.epochs, batch_size=args.batch_size, ckpt_interval=5)
