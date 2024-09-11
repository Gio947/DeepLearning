import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import random
import io

# Metodo usato per passare le immagini e se sono maschere per normalizzarle
def normalization(path, is_mask=False):
    # Estrae i dati byte dal dizionario
    image_bytes = path['bytes']
    # Converte i dati dell'immagine in un oggetto BytesIO
    image_io = io.BytesIO(image_bytes)
    # Carica l'immagine
    image = Image.open(image_io)
    # Se l'immagine è RGB, normalizza a valori tra 0 e 1
    if is_mask:
        image = image.point(lambda p: p * 255.0) #devo normalizzare in un altro modo, perchè i LiDAR sono coordinate e non colori

    return image

# Flip orizzontale dell'immagine
def applyHflip(lidar, image, mask):
    return TF.hflip(lidar), TF.hflip(image), TF.hflip(mask)
# Flip verticale dell'immagine
def applyVflip(lidar, image, mask):
    return TF.hflip(lidar), TF.vflip(image), TF.vflip(mask)
# Rotazione dell'immagine di 90 gradi
def applyRotate(lidar, image, mask, angle=90):
    return TF.rotate(lidar, angle), TF.rotate(image, angle), TF.rotate(mask, angle)

# Classe che estende Dataset per creare un dataset personalizzato, 
# dove vengono applicate delle trasformazioni alle immagini e poi trasformate in tensori
class MRIDataset(Dataset):
    def __init__(self, lidar_paths, image_paths, target_paths, train=True, dim_img=256):
        self.lidar_paths = lidar_paths.reset_index(drop=True)
        self.image_paths = image_paths.reset_index(drop=True)
        self.target_paths = target_paths.reset_index(drop=True)
        self.dim_img = dim_img

    def transform(self, lidar, image, mask):
        # Resize delle immagini e della maschera
        resize = transforms.Resize(size=(self.dim_img, self.dim_img))
        lidar = resize(lidar)
        image = resize(image)
        mask = resize(mask)

        # Trasformazioni che si possono applicare immagini
        augmentation_options = [
            applyHflip,
            applyVflip,
            applyRotate,
        ]

        # Applica una trasformazione casuale con una probabilità del 50%
        if random.random() > 0.5:
            transform_func = random.choice(augmentation_options)
            if transform_func == applyRotate:
                lidar, image, mask = transform_func(lidar, image, mask, angle=90)
            else:
                lidar, image, mask = transform_func(lidar, image, mask)

        # Trasformo le immagini e la maschera in tensori
        lidar = TF.to_tensor(lidar)
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        return lidar, image, mask

    def __getitem__(self, index):
        lidar_path = self.lidar_paths[index]
        img_path = self.image_paths[index]
        msk_path = self.target_paths[index]
        # Normalizzo la maschera in modo che 0,1 diventino 0 e 255 rispettivamente
        lidar = normalization(lidar_path, False)
        image = normalization(img_path, False)
        mask = normalization(msk_path, True)
        
        x_lidar, x_image, y = self.transform(lidar, image, mask)
        return x_lidar, x_image, y

    def __len__(self):
        return len(self.image_paths)