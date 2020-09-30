import torch
from tqdm import tqdm
def train_model(model,optimizer,epochs,criterion,trainloader,validation_loader=None,device='cuda',checkpoint=None):
    """

    Args:
        model(nn.Module):
        optimizer(nn.optim):
        criterion(nn.CrossEntropy)
        trainloader(torch.utils.data.Dataloader):
        validation_loader(torch.utils.data.Dataloader):
        device(torch.device):
        checkpoint(str): 
    """ 
    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        print('Starting training from checkpoint')
        #check how to load optimizer in checkpoint
        #Implement that later
    for epoch in tqdm(range(epochs)):
        
        model.train()
        running_loss = 0.0
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
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        print(f'Train: Epoch:{epoch}  Loss:{running_loss / total} Accuracy:{100 * correct / total }')

        model.eval()
        eval_loss = 0.0
        eval_correct = 0.0
        total = 0
        correct = 0.0
        for idx , mini_batches in tqdm(enumerate(validation_loader)):
            imgs , labels = mini_batches
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            output = model(imgs)
            _ , preds = torch.max(output,1)
            loss = criterion(output,labels)
            eval_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        print(f'Test: Epoch:{epoch}  Loss:{eval_loss / total}, Eval_Accuracy{100* correct/ total}')
        
        if(epoch%5==0):
            torch.save(model.state_dict(),f'model{epoch}.pth')

    return model