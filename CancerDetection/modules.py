import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Residual(nn.Module):
    def __init__(self, in_features, out_features):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, out_features, 3, padding=1)
        )
        self.fit_features = nn.Sequential()     # identity
        if in_features != out_features:
            self.fit_features = nn.Sequential(
                nn.InstanceNorm2d(in_features),
                nn.ReLU(),
                nn.Conv2d(in_features, out_features, 1, bias=False)     # bias=False : to maintain original character
            )

    def forward(self, x):
        return self.fit_features(x) + self.block(x)


class Encoder(nn.Module):
    def __init__(self, out_features=256):
        super(Encoder, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=(1, 1)),
            nn.Conv2d(64, 64, (5, 5), padding=(2, 2)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, (3, 3), padding=(1, 1), stride=(2, 2)),
            nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(2, 2)),
            nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
        )

        self.residuals = nn.Sequential(
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, out_features),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.convolution(x)
        x = self.residuals(self.dropout(x))
        return x


class Classifier(nn.Module):
    def __init__(self, in_features):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


class EncodeClassifier(nn.Module):
    def __init__(self, imgsize):
        super(EncodeClassifier, self).__init__()
        out_features = 256
        self.encoder = Encoder(out_features)
        flattened_size = int(imgsize*imgsize / 16) * out_features
        self.classifier = Classifier(flattened_size)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

