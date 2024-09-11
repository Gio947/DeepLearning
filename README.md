# DeepLearning
Progetto del corso di DLAGM

In ogni cartella sono presenti i file del progetto:
- fileLiDAR/RGB/RGB_LiDAR.py, che è il file main che esegue il train o testing
- MRIDataset.py, che permette di caricare il dataset
- Performance.py, il file che ha al suo interno i metodi per calcolare la loss e le performance
- Subset.py, che suddivide il dataset in train,validation e test set
- UNet.py, file che al suo interno ha la struttura del modello UNet
E la cartella /runs che contiene alcuni eventi generati per Tensorboard

Oltre al project assignment è presente anche un file word con i risultati dei vari training eseguiti con diversi parametri dal nome 'RisultatiFinali.docx'

Per poter eseguire i file bisogna lanciare tramite il seguente comando il file main insieme al comando 'train' o 'test':
- Per RGB: python ./fileRGB.py [train|test]
- Per LiDAR: python ./fileLiDAR.py [train|test]
- Per RGB e LiDAR: python ./fileRGBLiDAR.py [train|test]


Cartella drive con i migliori pesi relativi ai tre task e il dataset composto da file .parquet : https://drive.google.com/drive/folders/1ArN0oiGm1_k3s2REy9MvaLSK6_2Mwjex?usp=drive_link
