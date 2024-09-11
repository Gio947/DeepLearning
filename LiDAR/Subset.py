from torch.utils.data import DataLoader
import pandas as pd
from MRIDataset import MRIDataset

# Classe che restituisce i dataloader per il training, validation e test set
class Subset(object):
    def __init__(self, name_dataset, bs_train=32, bs_test=16, bs_final_test=16, dim_img=256):
        self.name_dataset = name_dataset
        self.bs_train = bs_train
        self.bs_test = bs_test
        self.bs_final_test = bs_final_test
        self.dim_img = dim_img

        self.df_training_list = []

        # Leggo tutti i file .parquet che compongono il dataset
        for name in self.name_dataset:
            df = pd.read_parquet('../dataset/' + name + '.parquet')
            self.df_training_list.append(df)
        
        self.df_training = pd.concat(self.df_training_list)
        
        #------------------- Split in training, validation and test set in 70:15:15 ---------------------
        ds_split = len(self.df_training)//10

        total_len = len(self.df_training)
        train_len = total_len * 70 // 100
        val_len = total_len * 15 // 100
        test_len = total_len - train_len - val_len

        img_files_train, mask_files_train = self.df_training['lidar'][:train_len], self.df_training['mask'][:train_len]
        img_files_test, mask_files_test = self.df_training['lidar'][train_len:train_len+val_len], self.df_training['mask'][train_len:train_len+val_len]
        img_files_final_test, mask_files_final_test = self.df_training['lidar'][train_len+val_len:], self.df_training['mask'][train_len+val_len:]

        print(len(img_files_train), len(img_files_test), len(img_files_final_test))

        # Datasets
        self.train_dataset = MRIDataset(img_files_train, mask_files_train, self.dim_img)
        self.test_dataset = MRIDataset(img_files_test, mask_files_test, self.dim_img)
        self.final_test_dataset = MRIDataset(img_files_final_test, mask_files_final_test, self.dim_img)

        # Dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.bs_train, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.bs_test, shuffle=True)
        self.final_test_dataloader = DataLoader(self.final_test_dataset, batch_size=self.bs_final_test, shuffle=True)
        
    def get_dataloaders(self):
        return self.train_dataloader, self.test_dataloader, self.final_test_dataloader