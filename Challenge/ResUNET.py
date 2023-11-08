import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

# Modelos NN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




class Conv_3_k(nn.Module):
    def __init__(self, channels_in, channels_out ):
        super().__init__()
        #self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=3,  stride=1,padding=1)  # de movida va conv 1d
    def forward(self, x):
        #x = self.dropout(x)
        return self.conv1(x)

# class residual_block(nn.Module):
#
#
#     def __init__(self, channels_in, channels_out,   stride = 1):
#         super().__init__()
#
#         self.conv1 = Conv_3_k(channels_in, channels_out, stride)
#         self.bn1 = nn.BatchNorm1d(channels_out)
#         self.conv2 = Conv_3_k(channels_out, channels_out, stride=1)
#         self.bn2 = nn.BatchNorm1d(channels_out)
#
#         self.residual = nn.Sequential(nn.Conv1d(channels_in,channels_out, kernel_size = 1, stride = stride),
#                                           nn.BatchNorm1d(channels_out))
#     def forward(self,x):
#         identity = self.residual(x)
#         y = F.relu(self.bn1(self.conv1(x)))
#         y = self.bn2(self.conv2(y))
#
#         y += identity
#
#         return F.relu(y)




# Se hace el bloque de dos convoluciones una detras de la otra
class First_Conv(nn.Module):
    '''
    Double convolution block for U-Net
    '''
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels_in, channels_out,kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.Conv1d(channels_out, channels_out,kernel_size = 3, stride=1, padding=1),
        )
        self.residual = nn.Sequential(nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm1d(channels_out))
        #self.residual = Conv_3_k(channels_in, channels_out)

    def forward(self, x):
        #print('First_Conv x ' + str(x.size()))
        y = self.first_conv(x)
        #print('First_Conv y ' + str(y.size()))
        s = self.residual(x)
        #print('First_Conv s ' + str(s.size()))
        skip = y + s
        #print('First_Conv skip ' + str(skip.size()))
        return skip

class Double_Conv(nn.Module):
    '''
    Double convolution block for U-Net
    '''

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm1d(channels_in),
            nn.ReLU(),
            nn.Conv1d(channels_in, channels_out,kernel_size = 3, stride=2 , padding=1),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.Conv1d(channels_out, channels_out,kernel_size = 3, stride=1, padding=1),
        )
        #self.residual = nn.Sequential(nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=stride),
                                     #nn.BatchNorm1d(channels_out))
        self.residual = nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        #print('Double_Conv x ' + str(x.size()))
        y = self.double_conv(x)
        #print('Double_Conv y ' + str(y.size()))
        s = self.residual(x)
        #print('Double_Conv s ' + str(s.size()))
        skip = y + s
        #print('Double_Conv skip ' + str(skip.size()))
        return skip


# aca se hace el maxpooling para bajar y volver a hacer la doble convolucion

class Down_Conv(nn.Module):
    '''
    Down convolution part
    '''

    def __init__(self, channels_in, channels_out,stride):
        super().__init__()
        self.encoder = nn.Sequential(
            #nn.MaxPool1d(2, 2),
            #nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=2),
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
        self.upsample_layer = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest') , # interpola y luego hace convolucion de 1x1
            nn.Conv1d(channels_in, channels_in // 2, kernel_size=1, stride=1)
        )
        self.double_conv = nn.Sequential(
            nn.BatchNorm1d(channels_in),
            nn.ReLU(),
            nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.Conv1d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
        )
        # self.residual = nn.Sequential(nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=stride),
        # nn.BatchNorm1d(channels_out))
        #self.residual = nn.Conv1d(channels_in,  channels_in // 2, kernel_size=1, stride=1, padding=0)
        self.residual = nn.Conv1d(channels_out, channels_out, kernel_size=1, stride=1, padding=0)


        #self.residual = nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=2, padding=0)
        #self.decoder = Double_Conv(channels_in, channels_out)
        #self.residual = nn.Conv1d(channels_in, channels_in // 2, kernel_size=1, stride=2, padding=0)

    def forward(self, x1, x2):
        '''
        x1 - upsampled volume
        x2 - volume from down sample to concatenate
        '''
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)  # concantena a lo largo de la dimension de los canales
        x= self.double_conv(x)
        s = self.residual(x)
        skip = x + s
        return skip

        # x = self.residual(x)
        # return x


# aca se hace el modelo


class ResUNET(nn.Module):
    '''
    UNET model
    '''

    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        self.first_conv = First_Conv(channels_in, channels)     # 64, 1024
        self.down_conv1 = Down_Conv(channels, 2 * channels,stride =2)      # 128, 512
        self.down_conv2 = Down_Conv(2 * channels, 4 * channels,stride = 2)  # 256, 256
        self.down_conv3 = Down_Conv(4 * channels, 8 * channels,stride = 2)  # 512, 128

        self.middle_conv = Down_Conv(8 * channels, 16 * channels,stride =2)  # 1024, 64

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


# Prueba de dimensiones

def test():
    x = torch.randn((8, 1, 1024))
    model = ResUNET(1, 64, 3)
    return model(x)