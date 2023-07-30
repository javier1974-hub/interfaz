import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import os
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#import streamlit as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# bloque con constantes, ya que usa kernel_size=3  , stride=1 y poadding=1 en todos los casos
# padding =same es para que queden las señales de salida del mismo tamaño que las de entrada
# ahora cada vez que se use un bloque de convolucion solo se le pasa cantidad de canales de entrada y cantidad de canales de salida
# lo demas ya queda implicito
class Conv_3_k(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=1,
                               padding='same')  # de movida va conv 1d

    def forward(self, x):
        return self.conv1(x)


# Se hace el bloque de dos convoluciones una detras de la otra

class Double_Conv(nn.Module):
    '''
    Double convolution block for U-Net
    '''

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv_3_k(channels_in, channels_out),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),

            Conv_3_k(channels_out, channels_out),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


# aca se hace el maxpoolin para bajar y volver a hacer la doble convolucion

class Down_Conv(nn.Module):
    '''
    Down convolution part
    '''

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool1d(2, 2),
            Double_Conv(channels_in, channels_out)
        )

    def forward(self, x):
        return self.encoder(x)


# aca se hace la interpolacion para subir y volver a hacer la doble convolucion

class Up_Conv(nn.Module):
    '''
    Up convolution part
    '''

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            # nn.MaxPool1d(2, 2),
            nn.Upsample(scale_factor=2, mode='linear'),  # interpola y luego hace convolucion de 1x1
            nn.Conv1d(channels_in, channels_in // 2, kernel_size=1, stride=1)
        )
        self.decoder = Double_Conv(channels_in, channels_out)

    def forward(self, x1, x2):
        '''
        x1 - upsampled volume
        x2 - volume from down sample to concatenate
        '''
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)  # concantena a lo largo de la dimension de los canales
        return self.decoder(x)


# aca se hace el modelo


class UNET(nn.Module):
    '''
    UNET model
    '''

    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        self.first_conv = Double_Conv(channels_in, channels)  # 64, 1024
        self.down_conv1 = Down_Conv(channels, 2 * channels)  # 128, 512
        self.down_conv2 = Down_Conv(2 * channels, 4 * channels)  # 256, 256
        self.down_conv3 = Down_Conv(4 * channels, 8 * channels)  # 512, 128

        self.middle_conv = Down_Conv(8 * channels, 16 * channels)  # 1024, 64

        self.up_conv1 = Up_Conv(16 * channels, 8 * channels)
        self.up_conv2 = Up_Conv(8 * channels, 4 * channels)
        self.up_conv3 = Up_Conv(4 * channels, 2 * channels)
        self.up_conv4 = Up_Conv(2 * channels, channels)

        self.last_conv = nn.Conv1d(channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)

        x5 = self.middle_conv(x4)

        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)
        n = self.last_conv(u4)

        return n


def test():
    x = torch.randn((8, 1, 1024))
    model = UNET(1, 64, 3)
    return model(x)


preds = test()
print(preds.shape)

preds = test()
# st.write(preds.shape)

model_path = './model_20_7_23.pth'

# Cargo el modelo previamente guardado

modelo = UNET(1, 64, 3)
modelo.load_state_dict(torch.load(model_path))
modelo.eval()


# Pruebas de validacion del modelo entrenado
def plot_salida(pcg, preds):
    x = np.linspace(0, 1, 1024)
    z = np.squeeze(preds)
    y = np.squeeze(pcg)

    cmap = ListedColormap(['r', 'g', 'b'])
    norm = BoundaryNorm([0., 0.99, 1.1, 2], cmap.N)

    points = np.array([x, y], dtype=object).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(2)

    fig1 = plt.figure(figsize=(15, 15))
    plt.gca().add_collection(lc)

    plt.text(0.1, 0.1, 'Linea de Base', color='r', fontsize=22)
    plt.text(0.1, 0.13, 'S1', color='g', fontsize=22)
    plt.text(0.1, 0.16, 'S2', color='b', fontsize=22)

    # plt.axis('On')
    # plt.legend()
    plt.title('Segmentacion')

    plt.savefig('Segmentacion.png', dpi=800)
    #st.pyplot(fig1)


#uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    pcg = np.genfromtxt(uploaded_file, dtype=float, delimiter=',')  # abro el .csv
    pcg = torch.from_numpy(pcg)
    pcg = pcg.unsqueeze(0)
    # st.write(pcg.shape)
    torch.transpose(pcg, 0, 1)
    # st.write(pcg.shape)
    pcg_max = pcg.max().item()  # normaliza el pcg ya que van de -1  a 1
    pcg_min = pcg.min().item()
    pcg = (pcg - pcg_min) / (pcg_max - pcg_min)  # ahora queda entre 0 y 1
    pcg = pcg[None, :]
    # st.write(pcg.shape)
    pcg = pcg.to(device, dtype=torch.float32)  # pongo seniales en gpu
    modelo = modelo.to(device)  # mando el modelo a GPU
    with torch.no_grad():
        scores = modelo(
            pcg)  # calculo scores (tienen dos canales con la pribabilidad que pertenezca a una u otra clase)
        preds = torch.argmax(scores,
                             dim=1).float()  # calcula predicciones (como se queda con el mayor en la dimension de canales, da la mascara)

    pcg = pcg.squeeze(1).cpu().numpy()
    preds = preds.cpu().numpy()
    plot_salida(pcg, preds)


