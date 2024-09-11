from torch.utils.data import DataLoader
import pandas as pd
from MRIDataset import MRIDataset

# Classe che restituisce i dataloader per il training, validation e test set
class Subset(object):
    def __init__(self, name_training, name_testing, name_validation, bs_train=32, bs_test=16, bs_final_test=16, dim_img=256):
        self.name_training = name_training
        self.name_testing = name_testing
        self.name_validation = name_validation
        self.bs_train = bs_train
        self.bs_test = bs_test
        self.bs_final_test = bs_final_test
        self.dim_img = dim_img

        self.df_training_list = []
        self.df_testing_list = []
        self.df_validation_list = []

        # Leggo tutti i file .parquet che compongono il training set
        for name in self.name_training:
            df = pd.read_parquet('../dataset/' + name + '.parquet')
            self.df_training_list.append(df)
        # Leggo tutti i file .parquet che compongono il test set
        for name in self.name_testing:
            df = pd.read_parquet('../dataset/' + name + '.parquet')
            self.df_testing_list.append(df)
        # Leggo tutti i file .parquet che compongono il validation set
        for name in self.name_validation:
            df = pd.read_parquet('../dataset/' + name + '.parquet')
            self.df_validation_list.append(df)

        self.df_training = pd.concat(self.df_training_list)
        self.df_testing = pd.concat(self.df_testing_list)
        self.df_validation = pd.concat(self.df_validation_list)

        print("Training set : ", self.df_training.info())
        print("Validation set : ", self.df_validation.info())
        print("Testing set : ", self.df_testing.info())
        
        #------------------- Split in training, validation and test set ---------------------
        #Datasets
        self.train_dataset = MRIDataset(self.df_training['lidar'], self.df_training['image'] , self.df_training['mask'], self.dim_img)
        self.test_dataset = MRIDataset(self.df_validation['lidar'], self.df_validation['image'] , self.df_validation['mask'], self.dim_img)
        self.final_test_dataset = MRIDataset(self.df_testing['lidar'], self.df_testing['image'], self.df_testing['mask'], self.dim_img)

        # Dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.bs_train, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.bs_test, shuffle=True)
        self.final_test_dataloader = DataLoader(self.final_test_dataset, batch_size=self.bs_final_test, shuffle=True)

    def get_dataloaders(self):
        return self.train_dataloader, self.test_dataloader, self.final_test_dataloader