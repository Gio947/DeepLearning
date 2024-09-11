import torch
import torch.nn as nn

def conv_layer(input_channels, output_channels, use_dropout=False):
    layers = [
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU()
    ]
    # Dropout con un rate di 0.5
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # In ingresso 3 in quanto è RGB e quindi ha 3 canali
        self.down_1 = conv_layer(3, 64) #128x128
        self.down_2 = conv_layer(64, 128) #64x64
        self.down_3 = conv_layer(128, 256) #32x32
        self.down_4 = conv_layer(256, 512, use_dropout=True) #16x16
        self.down_5 = conv_layer(512, 1024, use_dropout=True) #8x8
        
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = conv_layer(1024, 512, use_dropout=True)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = conv_layer(512, 256)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = conv_layer(256, 128)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = conv_layer(128, 64)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0) #ultimo livello converte in un solo canale finale
        self.output_activation = nn.Sigmoid()
        
                
    def forward(self, img):
        x1 = self.down_1(img)
        x2 = self.max_pool(x1)
        x3 = self.down_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_4(x6)
        x8 = self.max_pool(x7) 
        x9 = self.down_5(x8)
        
        x = self.up_1(x9) 
        x = self.up_conv_1(torch.cat([x, x7], 1))
        x = self.up_2(x)
        x = self.up_conv_2(torch.cat([x, x5], 1)) 
        x = self.up_3(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))
        x = self.up_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))
        
        x = self.output(x)
        x = self.output_activation(x)
        
        
        return x