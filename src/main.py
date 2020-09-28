from dataset.dataset import CustomDataset 
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ ==  '__main__':

    transform = transforms.Compose([transforms.ToPILImage(),\
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4601,0.4601,0.4601],[0.2701,0.2701,0.2701])
        ])
    dataset = CustomDataset('/home/ncai/Dataset/Distracted driver dataset/v1_cam1_no_split',\
        '/home/ncai/Dataset/Distracted driver dataset/v1_cam1_no_split/Train_data_list.csv',\
        transform=transform)
    
    train_loader = DataLoader(dataset,batch_size=16,shuffle=True)
