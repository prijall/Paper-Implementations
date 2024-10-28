import torch, os, config, numpy as np
from tqdm import tqdm
from datetime import datetime
import YOLOv1_Scratch
from data import YoloPascalVocDataset
from loss import SumSquaredErrorLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.autograd.set_detect_anamoly(True)
    writer=SummaryWriter()
    now=datetime.now()

    model=YOLOv1_Scratch.to(device)
    loss_function=SumSquaredErrorLoss()

    optimizer= torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    #loading the dataset
    train_set=YoloPascalVocDataset('train', normalize=True, augment=True)
    test_set=YoloPascalVocDataset('test', normalize=True, augment=True)

    train_loader=DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True, 
        shuffle=True
    )

    test_loader=DataLoader(
        test_set, 
        batch_size= config.BATCH_SIZE,
        num_workers=8, 
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )

    # Folder Creation:
    root=os.path.join(
          'YOLOv1_Scratch',
          'yolo_v1',
          now.strftime('%m_%d_%y'),
          now.strftime('%H_%M_%S')
    )
    weight_dir=os.path.join(root, 'weights')
    if not os.path.isdir('weight_dir'):
        os.makedirs('weight_dir')

    # Metrics:
    train_losses=np.empty((2, 0))
    test_losses=np.empty((2, 0))
    train_errors=np.empty((2, 0))
    test_errors=np.empty((2, 0))

    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'test_losses'), test_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'test_errors'), test_errors)


                #################
                #    Training   #
                #################

    for epoch in tqdm(range(config.WARMUP_EPOCHS + config.EPOCHS), desc='EPOCH'):
        model.train()
        train_loss=0
        for data, labels, _ in tqdm(train_loader, desc='Train', leave=False):
            data=data.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()
            prediction=model.forward(data)
            loss=loss_function(prediction, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            del data, labels
        
        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)

        if epoch % 4 == 0:
            model.eval()

            with torch.no_grad():
                test_loss=0
                for data, labels, _ in tqdm(train_loader, desc='Train', leave=False):
                   data=data.to(device)
                   labels=labels.to(device)

                   prediction=model.forward(data)
                   loss=loss_function(prediction, labels)

                   test_loss += loss.item() / len(test_loader)
                   del data, labels

            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)