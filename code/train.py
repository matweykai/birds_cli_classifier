import os
import shutil
from time import time
import cv2
from copy import deepcopy
import pickle

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score         

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch import optim, nn
from torch.optim import lr_scheduler
from torchvision import models

import albumentations as A
import albumentations.pytorch

from utils import RANDOM_SEED, loader_gen, seed_worker, config_randomness
from preprocessing import preprocess_images_folder


class BirdsDataset(Dataset):
    """ PyTorch implementation of Birds dataset """
    def __init__(self, ds_path: str, cust_transforms=None):
        self.ds_path = ds_path
        # Store image path and its label
        self.imgs_path_list = [(os.path.join(self.ds_path, temp_folder, temp_img), temp_folder)
                               for temp_folder in os.listdir(self.ds_path) 
                               for temp_img in os.listdir(os.path.join(self.ds_path, temp_folder))]
        self.cust_transforms = cust_transforms

        trans = albumentations.pytorch.ToTensorV2()
        self.to_tensor = lambda x: trans(image=x)['image']
    
    def __getitem__(self, ind: int):
        """ Returns image by index """
        img_path, label = self.imgs_path_list[ind]
        
        # Image preprocessing
        raw_img = cv2.imread(img_path)
        prep_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        if self.cust_transforms:
            prep_img = self.cust_transforms(image=prep_img)['image']
        
        return self.to_tensor(prep_img), int(label)
    
    def __len__(self):
        """ Returns dataset length """
        return len(self.imgs_path_list)


def _train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, loss, epoch_num: int, scheduler=None):
    print('Training started!\n')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loss_history = list()
    val_loss_history = list()
    
    val_acc_history = list()
    val_metric_history = list()
    
    model.to(device)
    
    for epoch_ind in range(epoch_num):
        temp_epoch_loss = 0
        temp_val_loss = 0
        
        # Train model
        model.train()
        
        for img_batch, target_batch in train_loader:      
            img_batch = img_batch.type(torch.FloatTensor).to(device)
            target_batch = target_batch.type(torch.LongTensor).to(device)
            
            # Forward pass
            model_pred = model(img_batch)
            
            cur_loss = loss(model_pred, target_batch)
            
            # Backward pass
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
            
            #print(cur_loss)
            temp_epoch_loss += cur_loss.item()
            #print(temp_epoch_loss)
            torch.cuda.empty_cache()
            
        if scheduler:
            scheduler.step()
        # Save train loss    
        train_loss_history.append(temp_epoch_loss / len(train_loader))
        
        # Validate model
        model.eval()
        
        model_answers_list = list()
        model_probs_list = list()
        true_labels_list = list()
        
        for img_batch, target_batch in val_loader:
            img_batch = img_batch.type(torch.FloatTensor).to(device)
            target_batch = target_batch.type(torch.LongTensor).to(device)
            
            model_pred = model(img_batch).detach()
            
            val_loss = loss(model_pred, target_batch)
            
            temp_val_loss += val_loss.item()
            
            model_answers_list.extend(list(torch.argmax(model_pred, dim=1).cpu().numpy()))
            model_probs_list.extend(nn.functional.softmax(model_pred.cpu(), dim=1).numpy())
            true_labels_list.extend(list(target_batch.cpu().numpy()))
            
            torch.cuda.empty_cache()
        
        val_acc_history.append(accuracy_score(true_labels_list, model_answers_list))
        val_loss_history.append(temp_val_loss / len(val_loader))
        
        model_probs_matrix = model_probs_list[0]
        
        for item in model_probs_list[1:]:
            model_probs_matrix = np.vstack((model_probs_matrix, item))      
        
        val_metric_history.append(f1_score(true_labels_list, model_answers_list, average='macro'))
        
        print(f'------------ Epoch #{epoch_ind + 1} Train loss: {round(train_loss_history[-1], 6)} Val loss: {round(val_loss_history[-1], 6)} ' + \
              f'F1: {round(val_metric_history[-1], 4)} ACC: {round(val_acc_history[-1], 4)} DIFF: {round(train_loss_history[-1] - val_loss_history[-1], 4)}   ------------')
        
        print('\nTraining finished!')

def _preprocess_data(data_dir_path: str):
    print('Preprocessing started!\n')
    counter = 1

    for temp_folder in os.listdir(data_dir_path):
        folder_path = os.path.join(data_dir_path, temp_folder)

        if os.path.isdir(folder_path):
            preprocess_images_folder(folder_path)

        print(f"Processed #{counter}")
        counter += 1 

    print('\nPreprocessing finished!')


def train_VIT():
    preprocessed_dir_path = '/data/preprocessed'
    raw_data_dir_path = '/data/raw/'
    model_dir_path = '/model/'
    
    # Check if preprocessed image directory is empty 
    if os.path.exists(preprocessed_dir_path):
        for folder in os.listdir(preprocessed_dir_path):
            shutil.rmtree(os.path.join(preprocessed_dir_path, folder))
        _preprocess_data(raw_data_dir_path)

    config_randomness()

    try:
        with open('/model/base_model.bin', 'rb') as file:
            base_model = pickle.load(file)
    except FileNotFoundError:
        base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        # Cache model
        with open('/model/base_model.bin', 'wb') as file:
            pickle.dump(base_model, file)
        
    

    # Define parameters for training
    batch_size = 32
    epoch_num = 15
    lr = 0.001

    # Define augmentations
    train_transforms_list = A.Compose(
        [
            A.augmentations.geometric.resize.Resize(384, 384),
            A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HorizontalFlip(),
        ]
    )

    valid_transforms_list = A.Compose(
        [
            A.augmentations.geometric.resize.Resize(384, 384),
            A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Define dataset
    train_data = BirdsDataset(preprocessed_dir_path, train_transforms_list)

    # Create a stratified train test split
    ind_list = list()
    labels_list = list()

    for temp_ind in range(len(train_data)):
        ind_list.append(temp_ind)
        labels_list.append(train_data[temp_ind][1])
        
    train_ind_list, valid_ind_list = train_test_split(ind_list, test_size=0.3, stratify=labels_list, random_state=RANDOM_SEED)

    train_ds = Subset(train_data, train_ind_list)

    # Change transforms for validation set
    valid_dataset = deepcopy(train_data)
    valid_dataset.cust_transforms = valid_transforms_list

    valid_ds = Subset(valid_dataset, valid_ind_list)

    # Create sampler with data oversampling function
    uniq_labels_list, labels_count_list = np.unique(labels_list, return_counts=True)
    class_weights = {uniq_labels_list[temp_ind]: sum(labels_count_list) / labels_count_list[temp_ind] for temp_ind in range(len(uniq_labels_list))}

    train_ind_weight_list = [class_weights[labels_list[temp_ind]] for temp_ind in train_ind_list]

    w_train_sampler = WeightedRandomSampler(train_ind_weight_list, len(train_ind_weight_list), replacement=True, 
                                            generator=torch.Generator().manual_seed(RANDOM_SEED))

    # Define loss function
    loss = nn.CrossEntropyLoss()

    # Freeze backbone model weigths
    model = base_model
    for temp_param in model.parameters():
        temp_param.requires_grad = False

    # ViT
    model.heads.head = nn.Linear(model.heads.head.in_features, len(uniq_labels_list))
    model.heads.head.requires_grad = True

    # Define data loaders
    train_loader = DataLoader(train_ds, batch_size, num_workers=2, sampler=w_train_sampler, 
                            worker_init_fn=seed_worker, generator=loader_gen)
    val_loader = DataLoader(valid_ds, batch_size, shuffle=False, num_workers=2)    


    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    # Start training
    _train(model, train_loader, val_loader, optimizer, loss, epoch_num, scheduler)

    # Save model
    with open(os.path.join(model_dir_path, f'model_{time()}'), 'wb') as file:
        pickle.dump(model, file)

    print('Model saved!')
