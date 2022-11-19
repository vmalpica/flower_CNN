import os
from this import d
import torch
import argparse
import random
import torch.nn as nn
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms


from flower_cnn.convnet import ConvNetwork
from flower_cnn.data import FlowerDataset


parser = argparse.ArgumentParser(description='Training the CNN network')
parser.add_argument(
    '--epochs', default=20, type=int, 
    help='total number of epochs'
)

parser.add_argument(
    '--resume', default='', type=str, 
    help='the path of the checkpoint'
)

root = os.path.dirname(__file__)

model_dir = os.path.join(root,'models')
data_dir = os.path.join(root,'dataset')


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def load_index(data_dir, test_split=0.2):
    dataset = {}
    dataset['train'] = {}
    dataset['test'] = {}
    train_idx = 0
    test_idx = 0

    index_path = os.path.join(f"{root}/dataset", data_dir.split('/')[-1]+'.json')
    train_filenames = random.sample(os.listdir(data_dir), int((1-test_split)*len(os.listdir(data_dir))))

    print("Loading indices from the dataset. . .")

    for fname in os.listdir(data_dir):
        if fname.endswith('.png') or fname.endswith('jpg'):
            if fname in train_filenames:
                dataset['train'][train_idx] = fname
                train_idx += 1

            else:
                dataset['test'][test_idx] = fname
                test_idx += 1

    with open(index_path, 'w') as fp:
          json.dump(dataset, fp)

    return index_path


# Define train function for one epoch
def train(loader, model, optimizer, criterion):
    loss_epoch = 0
    for batch_idx , [data, target] in enumerate(loader):
        data = data.to(device)  # 16x32x32
        target = target.to(device) #16

        # forward propagation
        scores = model(data) # model.forward(data) #16x20
        # print(scores)
        loss = criterion(scores, target)

        # back propagation
        optimizer.zero_grad()
        loss.backward()

        # Let the optimizer fit the model to the data
        optimizer.step()

        loss_epoch += loss.item()

        if batch_idx % 4 == 0:
            print(f"Step [{batch_idx}/{len(loader)}] ------> loss = {loss.item()}")

    return loss_epoch


def load_ckp(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['loss']

def save_ckp(state):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    epoch = state["epoch"]
    torch.save(state, f"{model_dir}/CNNmodel_epoch_{epoch}.pth")


def main():
    args = parser.parse_args()
    index_path = load_index(data_dir)

    # Hyperparameters

    batch_size = 16
    learning_rate = 0.001 
    n_classes = 20 
    n_epochs = args.epochs

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
                                
    train_dataset = FlowerDataset(
        root=data_dir,
        train=True,
        transform=norm_transform,
        index_path=index_path,
                                
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    test_dataset = FlowerDataset(
        root=data_dir,
        train=False,
        transform=norm_transform,
        index_path=index_path,
                                
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = ConvNetwork(in_channels=3, n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 500, eta_min = 1e-7)

    if args.resume != '':
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint {args.resume} . . .")
            model, optimizer, scheduler, epoch, loss_log = load_ckp(args.resume, model, optimizer, scheduler)
        else:
            print(f"Checkpoint file does not exist!")
    else:
        epoch = 0
        loss_log = []

    best_loss = train(loader=train_loader, model=model, optimizer=optimizer, criterion=criterion)

    model.train()

    for ix in range(epoch+1, n_epochs + 1):
        print(f"------ Epoch {ix} ------")
        loss_epoch = train(loader=train_loader, model=model, optimizer=optimizer, criterion=criterion)
        loss_log.append(loss_epoch)
        if loss_epoch < best_loss and ix % 5 == 0:
            best_loss = loss_epoch

            checkpoint = {
                'epoch': ix,
                'loss' : loss_log,
                'model': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
            }

            save_ckp(state=checkpoint)
        scheduler.step()

if __name__ == '__main__':
    main()