from dataset.dataset import CustomDataset 
from models.net import ClassifierNet
from utils.train import train_model
from utils.config import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from utils.eval import eval_model
NUM_CLASSES = 10
NO_EPOCHS = 20

if __name__ ==  '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set_transform = transforms.Compose([transforms.ToPILImage(),\
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4601,0.4601,0.4601],[0.2701,0.2701,0.2701])
        ])
    
    train_dataset = CustomDataset(dataset_root_folder,\
        train_set_csv,\
        transform=train_set_transform)
    
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,pin_memory=True,num_workers=4)
    
    test_set_transform = transforms.Compose([transforms.ToPILImage(),\
        transforms.Resize((244,244)),
        transforms.ToTensor(),
        transforms.Normalize([0.4601,0.4601,0.4601],[0.2701,0.2701,0.2701])        
    ])
    
    test_dataset = CustomDataset(dataset_root_folder , \
        test_set_csv, transform = test_set_transform)
    
    test_loader = DataLoader(test_dataset , batch_size=8 , shuffle=True,pin_memory=True,num_workers=4)
    
    checkpoint = 'model_final.pth'

    model = ClassifierNet(NUM_CLASSES)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adagrad(model.parameters(),lr=0.001,weight_decay=1e-3)
    
    trained_model = train_model(model,optim,NO_EPOCHS,criterion,train_loader,validation_loader=test_loader,device= device)
    torch.save(trained_model.state_dict(),'model_final.pth')

    eval_model(model,checkpoint,test_loader,criterion,device)