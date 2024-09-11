import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from UNet import UNet
from Performance import Performance
from Subset import Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from itertools import islice
import argparse

# load image function
def pil_loader(path):
    
    image_bytes = path['bytes']
    
    image_io = io.BytesIO(image_bytes)
    
    image = Image.open(image_io)
    
    return image

#Settings and Hyperparameters

#Per inferenza e trai  con il prof usare cpu
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)


# Iperparametri: batch size, epoche, learning rate, dimensione immagine
bs_train, bs_test, bs_final_test = 8, 4, 4
lr = 0.001
dim_img = 256


""" rows,cols=3,3
fig=plt.figure(figsize=(10,10))
for i in range(1,rows*cols+1):
    fig.add_subplot(rows,cols,i)
    img_path=df['lidar'][i]
    msk_path=df['mask'][i]
    img=pil_loader(img_path)
    msk=pil_loader(msk_path)
    plt.imshow(img)
    plt.imshow(msk,alpha=0.4)
    plt.axis('off')
plt.show() """


# Inizializzazione pesi he_normal
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


#inizializzazione modello e dei pesi
model = UNet().to(device)
model.apply(init_weights)

#ottimizzatore Adam, con impostazione del learning rate e decadimento dei pesi per l'overfitting
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9,0.999), amsgrad=True)

#ottimizzatore SGD
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#scheduler per ridurre il learning rate se non si riduce l'errore per due epoche di fila
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

#array che contiene tutti i file del dataset
name_dataset = ['train-00000-of-00021-e8c1d8ec39d4e7f8', 'train-00001-of-00021-0fe57e09489e73fb',
                 'train-00002-of-00021-8607eec042661272', 'train-00003-of-00021-5a8103be59b83d86',
                 'train-00004-of-00021-13608d949031da48', 'train-00005-of-00021-6168fd6f11a87adb']

#realizzazione dei subset per il training, validation e testing
subset = Subset(name_dataset, bs_train, bs_test, bs_final_test, dim_img)
train_dataloader, test_dataloader, final_test_dataloader = subset.get_dataloaders()

#istanzio la classe dove ci sono i metodi per calcolare performance
p = Performance()

# Nome e percorso su cui salvare il modello
PATH = './modello/unet_lidar3.pth'

#istanzio il writer per salvare i dati su TensorBoard
writer = SummaryWriter('./runs/provaLiDAR')

from matplotlib.lines import Line2D

# Metodo per visualizzare l'istogramma dei gradienti su TensorBoard
def add_gradient_hist(net):
    ave_grads = [] 
    layers = []
    for n,p in net.named_parameters():
        if ("bias" not in n):
            layers.append(n)
            if p.requires_grad: 
                ave_grad = np.abs(p.grad.clone().detach().cpu().numpy()).mean()
            else:
                ave_grad = 0
            ave_grads.append(ave_grad)
        
    layers = [layers[i].replace(".weight", "") for i in range(len(layers))]
    
    fig = plt.figure(figsize=(12, 12))
    plt.bar(np.arange(len(ave_grads)), ave_grads, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    #plt.grid(True)
    plt.legend([Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    #plt.show()
    
    return fig

# Metodo che esegue il train del modello
def train(model, epochs):
    
    # Array per tenere traccia della train loss e validation loss medie per il training
    avg_train_losses = []
    avg_test_losses = []
    
    # Imposto le variabili per eseguire l'early stopping, con un patience di 3
    earlystopping = False
    patience = 3  # Imposta la pazienza desiderata
    epochs_no_improve = 0  # Conta le epoche senza miglioramento

    #Training loop
    for epoch in range(epochs):
        
        # Si salvano le loss di training e validation, score, IoU e BIoU per una singola epoca
        train_losses = []
        test_losses = []
        list_scores = []
        list_Iou = []
        list_Biou = []
       
        model.train()

        loop = tqdm(enumerate(train_dataloader), total = len(train_dataloader), leave = False)
        for batch, (images, targets) in loop:
            
            images = images.to(device)
            targets = targets.to(device) # the ground truth mask

            optimizer.zero_grad()
            pred = model(images)
            loss = p.dc_loss(pred, targets)
            loss.backward()
            
            # Eseguo il clipping dei pesi, per evitare l'esplodere dei gradienti
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Salvo su TensorBoad la training loss e l'istogramma dei gradienti
            writer.add_scalar('training loss',loss)
            
            writer.add_figure('gradients',
                            add_gradient_hist(model),
                            global_step=epoch)
            
            train_losses.append(loss.item())
            
            # Controllo le predizioni che esegue e salvo gli esempi, solo nel primo batch, su TensorBoard
            with torch.no_grad(): # Evita di calcolare i gradienti durante la valutazione
                if batch == 1:
                    model.eval()

                    (img, mask) = next(iter(test_dataloader))
                    img = img.to(device)
                    mask = mask.to(device)
                    mask = mask[0]
                    pred = model(img).detach()

                    # Converti le immagini in un formato compatibile con add_image
                    img_np = img.cpu().numpy()
                    if img_np.ndim == 4:  # Se l'immagine ha un batch dimensionale
                        img_np = img_np[0]  # Prendi il primo elemento del batch
                    img_grid = np.transpose(img_np, (1, 2, 0))  # Trasponi da (C, H, W) a (H, W, C)

                    mask_np = mask.cpu().numpy()
                    if mask_np.ndim == 3:  # Se la maschera ha un batch dimensionale
                        mask_np = mask_np[0]  # Prendi il primo elemento del batch
                    mask_grid = np.squeeze(mask_np)  # Rimuovi dimensioni singolari

                    pred_np = pred.cpu().numpy()
                    if pred_np.ndim == 4:  # Se la predizione ha un batch dimensionale
                        pred_np = pred_np[0]  # Prendi il primo elemento del batch
                    pred_grid = np.squeeze(pred_np) > 0.5  # Rimuovi dimensioni singolari e applica soglia
                    # Aggiungi le immagini a TensorBoard
                    writer.add_image('Original Image', img_grid, global_step=batch, dataformats='HWC')
                    writer.add_image('Original Mask', mask_grid, global_step=batch, dataformats='HW')
                    writer.add_image('Prediction', pred_grid, global_step=batch, dataformats='HW')

                    model.train()

        model.eval()
        
        # Calcolo la validation loss e le perfomance per tutti i batch dell'epoca
        with torch.no_grad():    
            for test_batch, (test_images, test_targets) in enumerate(test_dataloader):
                test_images = test_images.to(device)
                test_targets = test_targets.to(device)
                test_pred = model(test_images).detach()

                test_loss = p.dc_loss(test_pred, test_targets).item()
                test_losses.append(test_loss)
                
                preds = test_pred.squeeze(1).cpu().detach().numpy()
                mask = test_targets.squeeze(1).cpu().detach().numpy()
                score = p.calculate_score(preds, mask)
                list_scores.append(score['score'])
                list_Iou.append(score['iou'])
                list_Biou.append(score['biou'])

            # Calcola la media delle loss e delle performance per l'epoca
            epoch_avg_train_loss = np.average(train_losses)
            epoch_avg_test_loss = np.average(test_losses)
            
            epoch_avg_score = np.average(list_scores)
            epoch_avg_IoU = np.average(list_Iou)
            epoch_avg_BIoU = np.average(list_Biou)
            
            avg_train_losses.append(epoch_avg_train_loss)
            avg_test_losses.append(epoch_avg_test_loss)

            print_msg = (f'train_loss: {epoch_avg_train_loss:.5f} ' + f'valid_loss: {epoch_avg_test_loss:.5f} '
                         + f'score: {epoch_avg_score:.5f} ' + f'IoU: {epoch_avg_IoU:.5f} '
                         + f'BIoU: {epoch_avg_BIoU:.5f} ')
            
            # Salvo su TensorBoard le performance per l'epoca
            writer.add_scalar('IoU', epoch_avg_IoU, epoch)
            writer.add_scalar('BIoU', epoch_avg_BIoU, epoch)
            writer.add_scalar('Score', epoch_avg_score, epoch)

            print(print_msg)
            
        scheduler.step(epoch_avg_test_loss) # Aggiorno il learning rate , però dovrei farlo sul validation loss
        
        if epoch > 10:  # Early stopping con un minimo di 10 epoche
            if len(avg_test_losses) > 1 and avg_test_losses[-1] >= avg_test_losses[-2]:  # Controlla se la loss non diminuisce
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early Stopping Triggered With Patience {patience}")
                    torch.save(model.state_dict(), PATH)
                    earlystopping = True
            else:
                epochs_no_improve = 0  # Resetta il conteggio se la loss migliora

        if earlystopping:
            break
        
    # Salvo il modello
    torch.save(model.state_dict(), PATH)

    return  model, avg_train_losses, avg_test_losses


def test(model, final_test_dataloader):
    
    # Carico il modello
    model.load_state_dict(torch.load(PATH))

    # Imposta il modello in modalità di valutazione
    model.eval()
    
    # Istanzio il writer per salvare i dati su TensorBoard (se non già istanziato)
    writer = SummaryWriter('./runs/test_images')

    # Itera su 3 immagini nel dataloader
    for i, (img, mask) in enumerate(islice(final_test_dataloader,3)):
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        
        # LiDAR Image
        writer.add_image(f'Test Original LiDAR {i+1}', img[0].cpu().numpy().squeeze(0), global_step=i, dataformats='HW')
        
        # Mask
        writer.add_image(f'Test Mask {i+1}', mask[0].cpu().numpy().squeeze(0), global_step=i, dataformats='HW')
        
        # Prediction
        prediction = pred.detach().cpu().numpy()
        if prediction.ndim == 4:  # Se la predizione ha un batch dimensionale
            prediction = prediction[0]  # Prendi il primo elemento del batch
        pred_grid = np.squeeze(prediction) > 0.5  # Rimuovi dimensioni singolari e applica soglia

        writer.add_image(f'Test Prediction {i+1}', pred_grid, global_step=i, dataformats='HW')
        
        
        # Visualizza l'immagine, la maschera e la predizione
        plt.figure(figsize=(12,12))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img[0].cpu().numpy()))
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow((mask[0].cpu().numpy()).transpose(1,2,0).squeeze(axis=2))
        plt.title('Original Mask')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(pred[0].cpu()) > .5)
        plt.title('Prediction')
        plt.axis('off')
        plt.show()

        # Calcola e stampa il punteggio per l'immagine corrente
        preds = pred.squeeze(1).cpu().detach().numpy()
        mask = mask.squeeze(1).cpu().detach().numpy()
        score = p.calculate_score(preds, mask)
        print('Score : ', score['score'])
        print('IoU : ', score['iou'])
        print('BIoU : ' , score['biou'])
        print('-------------------------------------------')
    
    writer.close()
        

def main():
    parser = argparse.ArgumentParser(description="Script per addestrare e testare il modello")
    parser.add_argument('operation', choices=['train', 'test'], help="Specifica se eseguire il training o il testing")
    args = parser.parse_args()

    if args.operation == 'train':
        
        epochs = 10
        # Esegui il training
        best_model, avg_train_losses, avg_val_losses = train(model, epochs)

        # Crea un array per le epoche
        epochs = range(1, len(avg_train_losses) + 1)

        # Crea il grafico
        plt.figure(figsize=(10,5))
        plt.plot(epochs, avg_train_losses, 'b', label='Perdite di addestramento')
        plt.plot(epochs, avg_val_losses, 'r', label='Perdite di validazione')
        plt.title('Perdite di addestramento e validazione')
        plt.xlabel('Epoche')
        plt.ylabel('Perdite')
        plt.legend()

        plt.savefig('graficoLiDAR.png', dpi=300, bbox_inches='tight')

    elif args.operation == 'test':
        # Esegui il testing
        test(model, final_test_dataloader)


if __name__ == "__main__":
    main()