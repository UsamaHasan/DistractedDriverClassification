import torch
from tqdm import tqdm
def train_model(model,optimizer,epochs,criterion,trainloader,validation_loader=None,device='cuda'):
    """

    Args:
        model(nn.Module):
        optimizer(nn.optim):
        criterion(nn.CrossEntropy)
        trainloader(torch.utils.data.Dataloader):
        device(torch.device): 
    """ 
    
    
    for epoch in tqdm(range(epochs)):
        
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        correct = 0.0
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
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        print(f'Train Accuracy:{100 * correct / total}')
        epoch_loss = running_loss / len(trainloader)
        epoch_acc =  running_correct.double() / len(trainloader)
        print(f'Train: Epoch{epoch}  Loss:{epoch_loss} ,  Accuracy{epoch_acc}')
        
        """
        model.eval()
        eval_loss = 0.0
        eval_correct = 0
        for idx , mini_batches in enumerate(validation_loader):
            imgs , labels = mini_batches
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            output = model(imgs)
            _ , preds = torch.max(output,1)
            loss = criterion(output,labels)
            eval_loss+= loss.item()
            eval_correct += torch.sum(preds == labels.data)
        print(f'Eval Loss:{eval_loss/len(validation_loader)}, Eval_Accuracy{eval_correct.double()/len(validation_loader)}')
        """
        if(epoch%5==0):
            torch.save(model.state_dict(),f'model{epoch}.pth')

    return model