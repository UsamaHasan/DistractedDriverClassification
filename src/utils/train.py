import torch
from tqdm import tqdm
def train_model(model,optimizer,epochs,criterion,trainloader,device):
    """

    Args:
        model(nn.Module):
        optimizer(nn.optim):
        criterion(nn.CrossEntropy)
        trainloader(torch.utils.data.Dataloader):
        device(torch.device): 
    """ 
    model.train()
    
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        running_correct = 0
        
        for idx , mini_batch in tqdm(enumerate(trainloader)):
            imgs , labels = mini_batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = labels.long()
            output = model(imgs)
            _ , preds = torch.max(output,1)
            optimizer.zero_grad()
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            running_correct += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(trainloader)
        epoch_acc =  running_correct.double() / len(trainloader)
        print(f'Train: Epoch{epoch}  Loss:{epoch_loss} ,  Accuracy{epoch_acc}')

    return model