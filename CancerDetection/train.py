import torch
from torch.utils.data import DataLoader
from .dataset import TrainDataset
import tensorboardX
from tqdm import tqdm
from .modules import EncodeClassifier
from torchvision.transforms import transforms
from .util import weights_init
import argparse
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--savepath', type=str, required=True)
parser.add_argument('--datapath', type=str, required=True)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--writeinterval', type=int, default=100)
parser.add_argument('--ckptinterval', type=int, default=10)
parser.add_argument('--numworkers', type=int, default=0)
parser.add_argument('--imgsize', type=int, default=96)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'torch device: {device}')

dataset = TrainDataset(transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]), datapath=args.datapath)
loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers)
model = EncodeClassifier(args.imgsize).apply(weights_init).to(device)

l2loss = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch//4, args.epoch//2, args.epoch//4*3])

writer = tensorboardX.SummaryWriter()

iteration = 0
loss_sum = 0
for epoch in range(1, args.epoch+1):
    for img, label in tqdm(loader):
        optimizer.zero_grad()

        x = img.to(device)
        y_label = label.to(device).view([-1, 1])

        y = model(x)
        loss = l2loss(y, y_label)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        iteration += 1
        if iteration % args.writeinterval == 0:
            writer.add_scalars('train', {
                'loss': loss_sum / args.writeinterval
            }, iteration)
            loss_sum = 0
    scheduler.step()
    if epoch % args.ckptinterval == 0:
        model.eval().cpu()
        torch.save({
            'state_dict': model.state_dict()
        }, path.join(args.savepath, 'ckpts', f'checkpoint{epoch}.pt'))
        model.to(device).train()

writer.close()
model.eval().cpu()
torch.save({
    'state_dict': model.state_dict()
}, path.join(args.savepath, 'model.pt'))
print('complete training')
