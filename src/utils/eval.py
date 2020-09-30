import torch

def eval_model(model,checkpoint,validation_loader,criterion,device):
    """
    """
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
    model.eval()    
    eval_loss = 0.0
    eval_correct = 0
    total = 0
    
    for idx , mini_batches in enumerate(validation_loader):
        imgs , labels = mini_batches
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        output = model(imgs)
        _ , preds = torch.max(output,1)
        loss = criterion(output,labels)
        eval_loss+= loss.item()
        eval_correct += (preds == labels).sum().item()
        total+=labels.size(0)

    print(f'Eval Loss:{eval_loss/total}, Eval_Accuracy{eval_correct/total}')
