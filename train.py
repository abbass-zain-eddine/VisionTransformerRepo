import torch,sys,os,time
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
import torchvision
import Models
import glob
import matplotlib.pyplot as plt
from torchvision import transforms

import random
from CustomDataSet import NoisyDataset
from torch.utils.tensorboard import SummaryWriter
plt.switch_backend('agg') # for servers not supporting display



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)


def im_class(label,classes):
    c=[ classes[i] for i in np.array(label.cpu()).argmax(axis=1)]
    return c
    

def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

def train(model:nn.Module,optimizer: torch.optim,loss_fn,train_Loader: DataLoader,val_loader:DataLoader,log_dir=".",log_interval=10,num_epochs: int =10):
    TB=SummaryWriter(log_dir)
    script_time = time.time()
    train_epoch_loss = []
    val_epoch_loss = []
    running_train_loss = []
    running_val_loss = []
    model.to(device)
    print('\nTRAINING...')
    epoch_train_start_time = time.time()
    model.train()
    
    for epoch in range(num_epochs):
        running_loss=0.0
        for batch_idx, (Images, targets) in enumerate(train_Loader):
            batch_start_time = time.time()
            Images.to(device)
            output=model(Images)
            optimizer.zero_grad()
            targets=torch.vstack(targets)
            targets=targets.transpose(0,1).float().to(device)
            loss=loss_fn(output,targets)
            running_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item()
            if (batch_idx + 1)%10 == 0:
                batch_time = time.time() - batch_start_time
                m,s = divmod(batch_time, 60)
                print('train loss @batch_idx {}/{}: {} in {} mins {} secs (per batch)'.format(str(batch_idx+1).zfill(len(str(len(train_Loader)))), len(train_Loader), loss.item(), int(m), round(s, 2)))

            if((batch_idx-1) % log_interval == 0):             
                TB.add_images("images", Images[0], global_step=epoch*len(train_Loader)+batch_idx, dataformats='CHW')
                #TB.add_text("class", str(im_class(targets,["Speckle_Noise","Salt_Pepper","Uneven_Illumination"])), global_step=epoch*len(train_Loader)+batch_idx)
                true_label_name=str(im_class(targets.detach().clone(),["Speckle_Noise","Salt_Pepper","Uneven_Illumination"]))
                predicted_label_name=str(im_class(output.detach().clone(),["Speckle_Noise","Salt_Pepper","Uneven_Illumination"]))
                label_text = f'True label: {true_label_name[0]}\nPredicted label: {predicted_label_name[0]}'
                TB.add_text(f'labels_{batch_idx}', label_text)


        train_epoch_loss.append(np.array(running_train_loss).mean())
        TB.add_scalar("Loss/epoch", np.array(running_train_loss).mean(), epoch)
        epoch_train_time = time.time() - epoch_train_start_time
        m,s = divmod(epoch_train_time, 60)
        h,m = divmod(m, 60)
        print('\nepoch train time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))

        
        print('\nVALIDATION...')
        epoch_val_start_time = time.time()
        model.eval()
        with torch.no_grad():
            for batch_idx, (Images,targets) in enumerate(val_loader):
                Images = Images.to(device)
                output = model(Images)
                targets=torch.vstack(targets)
                targets=targets.transpose(0,1).float().to(device)
                loss = loss_fn(output, targets)
                running_val_loss.append(loss.item())

                if((batch_idx-1) % 10 == 0):
                    TB.add_images("images", Images[0], global_step=epoch*len(train_Loader)+batch_idx, dataformats='CHW')
                    #TB.add_text("class", str(im_class(targets,["Speckle_Noise","Salt_Pepper","Uneven_Illumination"])), global_step=epoch*len(train_Loader)+batch_idx)
                    true_label_name=str(im_class(targets.detach().clone(),["Speckle_Noise","Salt_Pepper","Uneven_Illumination"]))
                    predicted_label_name=str(im_class(output.detach().clone(),["Speckle_Noise","Salt_Pepper","Uneven_Illumination"]))
                    label_text = f'True label: {true_label_name[0]}\nPredicted label: {predicted_label_name[0]}'
                    TB.add_text(f'labels_{batch_idx}', label_text)
                if (batch_idx + 1)%log_interval == 0:
                    print('val loss   @batch_idx {}/{}: {}'.format(str(batch_idx+1).zfill(len(str(len(val_loader)))), len(val_loader), loss.item()))

        val_epoch_loss.append(np.array(running_val_loss).mean())

        epoch_val_time = time.time() - epoch_val_start_time
        m,s = divmod(epoch_val_time, 60)
        h,m = divmod(m, 60)
        print('\nepoch val   time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))

        train_loss=running_loss/len(train_Loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
    
    total_script_time = time.time() - script_time
    m, s = divmod(total_script_time, 60)
    h, m = divmod(m, 60)
    print(f'\ntotal time taken for running this script: {int(h)} hrs {int(m)} mins {int(s)} secs')
    
    print('\nFin.')

if __name__ == "__main__":
    model=Models.VisionTransformer(n_patches=16,token_dim=64,nbr_blocks=10,nbr_heads=4,output_dim=3)
    data_path="/home/zeineddine/ICIPCompet/ICIP New Data/Noises/"
    classes_directory=glob.glob(data_path+"*")
    images_path_train=[]
    images_path_val=[]

    for path in classes_directory:
        temp_paths=glob.glob(path+"/train/*.jpg")
        images_path_train.extend(temp_paths[:int(0.8*len(temp_paths))])
        images_path_val.extend(temp_paths[int(0.8*len(temp_paths)):])
    
    random.shuffle(images_path_train)
    random.shuffle(images_path_val)

    transform = transforms.Compose([ transforms.ToTensor(),transforms.Resize((640,640),transforms.InterpolationMode.BICUBIC)])
   
    print(classes_directory)
    train_ds=NoisyDataset(images_path_train,transform,Test=True)
    val_ds=NoisyDataset(images_path_val,transform,Test=True)
    train_loader=DataLoader(train_ds,batch_size=32)#,collate_fn=train_ds.collate_fn)
    val_loader=DataLoader(val_ds,batch_size=32)#,collate_fn=train_ds.collate_fn)
    train(model,torch.optim.Adam(model.parameters(),lr=1e-4),nn.CrossEntropyLoss(),train_loader,val_loader,".",log_interval=5,num_epochs=50)