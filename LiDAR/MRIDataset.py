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
    # Se l'immagine non è RGB, normalizza a valori tra 0 e 1
    if is_mask:
        image = image.point(lambda p: p * 255.0)

    return image

# Flip orizzontale dell'immagine
def applyHflip(image, mask):
    return TF.hflip(image), TF.hflip(mask)
# Flip verticale dell'immagine
def applyVflip(image, mask):
    return TF.vflip(image), TF.vflip(mask)
# Rotazione dell'immagine di 90 gradi
def applyRotate(image, mask, angle=90):
    return TF.rotate(image, angle), TF.rotate(mask, angle)

# Classe che estende Dataset per creare un dataset personalizzato, 
# dove vengono applicate delle trasformazioni alle immagini e poi trasformate in tensori
class MRIDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True, dim_img=256):
        self.image_paths = image_paths.reset_index(drop=True)
        self.target_paths = target_paths.reset_index(drop=True)
        self.dim_img = dim_img

    def transform(self, image, mask):
        # Resize dell'immagine e della maschera
        resize = transforms.Resize(size=(self.dim_img, self.dim_img))
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
                image, mask = transform_func(image, mask, angle=90)
            else:
                image, mask = transform_func(image, mask)
            
        # Trasformo l'immagine e la maschera in tensori
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        return image, mask

    def __getitem__(self, index):
        
        img_path = self.image_paths[index]
        msk_path = self.target_paths[index]
        # Normalizzo la maschera in modo che 0,1 diventino 0 e 255 rispettivamente
        image = normalization(img_path, False)
        mask = normalization(msk_path, True)
        
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)