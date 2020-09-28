from dataset.dataset import CustomDataset 
from models.net import ClassifierNet
from train import train
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models

NUM_CLASSES = 10
NO_EPOCHS = 100

if __name__ ==  '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    transform = transforms.Compose([transforms.ToPILImage(),\
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4601,0.4601,0.4601],[0.2701,0.2701,0.2701])
        ])
    dataset = CustomDataset('/home/ncai/Dataset/Distracted driver dataset/v1_cam1_no_split',\
        '/home/ncai/Dataset/Distracted driver dataset/v1_cam1_no_split/Train_data_list.csv',\
        transform=transform)
    
    train_loader = DataLoader(dataset,batch_size=16,shuffle=True)
    model = ClassifierNet(NUM_CLASSES)
    model = model.to(device)
    optim = torch.optim.Adagrad(model.parameters(),lr=0.001,weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    train(model,optim,NO_EPOCHS,criterion,train_loader,device)