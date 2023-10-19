import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#PATH = '/Algoritmos/Prueba_Torch_PyCharm/'
#TRAIN_PATH = '/Algoritmos/Prueba_Torch_PyCharm/train/'
#TRAIN_MASKS_PATH = '/Algoritmos/Prueba_Torch_PyCharm/train_masks/'
#TEST_PATH = '/Algoritmos/Prueba_Torch_PyCharm/test/'


PATH = 'D:/Algoritmos/Interfaz_Grafica/Challenge/'
TRAIN_PATH = 'D:/Algoritmos/Interfaz_Grafica/Challenge/train/'
TRAIN_MASKS_PATH = 'D:/Algoritmos/Interfaz_Grafica/Challenge/train_masks/'
TEST_PATH = 'D:/Algoritmos/Interfaz_Grafica/Challenge/test/'


# creating our own Dataset
# Se hereda de la clase Dataset de Pytorch
class PCG_Dataset(Dataset):
    def __init__(self, data, mask=None, pcg_transforms=None, mask_transforms=None):
        '''
        data - train data path (Son rutas )
        masks - train masks path (son rutas )
        '''
        self.train_data = data  # guarda rutas en variables  "x" phono
        self.train_masks = mask  # guarda rutas en variables "y" mascara 1


        self.pcg_transforms = pcg_transforms  # para data augmentation. Se usa para modificar la long de las seniales
        self.mask_transforms = mask_transforms  # ademas convierte a tensores de pytorch


        self.pcgs = sorted(os.listdir(self.train_data))  # son todas las seniales del directroria de train_data, ademas esta ordenada
        self.masks = sorted(os.listdir(self.train_masks)) # son todas las mascaras del directroria de train_masks, ademas esta ordenada



    def __len__(self):
        if (self.train_masks) is not None:
            assert (len(self.pcgs)) == len(self.masks), 'not the same number of images and masks'
        return len(self.pcgs)

    # regresa un elemento del dataset en la posicion del idx
    # tambien permite aplicar las transformaciones que necesitemos en el dataset

    def __getitem__(self, idx):
        pcg_name = os.path.join(self.train_data,self.pcgs[idx])  # concatena la ruta con el nombre de la imagen en la posicion idx

        pcg = np.genfromtxt(pcg_name, dtype=float, delimiter=',')  # abro el .csv

        trans = torch.from_numpy(pcg)  # convierto a tensor

        if self.pcg_transforms is not None:  # si hay transoformaciones, las aplica y convierte a tensor
            pcg = self.pcg_transforms(pcg)
        else:
            pcg = torch.from_numpy(pcg)


        pcg_max = pcg.max().item()                   # normaliza el pcg ya que van de -1  a 1
        pcg_min = pcg.min().item()
        pcg = (pcg - pcg_min) / (pcg_max - pcg_min)  # ahora queda entre 0 y 1
        pcg = pcg[None,:]

        

        if self.train_masks is not None:                            # chequea si hay mascaras, si hay, las carga
            mask_name = os.path.join(self.train_masks, self.masks[idx])  
            mask = np.genfromtxt(mask_name, dtype=float, delimiter=',')

            if self.mask_transforms is not None:  # si hay transoformaciones, las aplica y convierte a tensor
                mask = self.mask_transforms(mask)
            else:
                mask = torch.from_numpy(mask)         # si no hay transformacion solo convierte a tensor
                mask = mask[None, :]
                #mask_max = mask.max().item()          #normaliza las masrecaras para que quede entre 0 y 1 (cosa que ya pasa, pero por las dudas)
                #mask /= mask_max

        else:
            return pcg  # si no existen las mascaras, devuelve la senial
             

        return pcg, mask  # devuelve senial y mascaras 


# crea el dataset: recibe ruta de entrenamiento, ruta de las mascaras para entrenamiento y las transformaciones.
full_dataset = PCG_Dataset(TRAIN_PATH, TRAIN_MASKS_PATH)




BATCH_SIZE = 8                                        # tamabio del batch
TRAIN_SIZE = int(len(full_dataset)*0.7)  -1              # el 80% del dataset lo usa para entrenamiento
VAL_SIZE =  int((len(full_dataset) - TRAIN_SIZE)/2)              # el 20% restante lo usa para validacion
TEST_SIZE = int((len(full_dataset) - TRAIN_SIZE)/2)  # len(full_dataset) - TRAIN_SIZE - VAL_SIZE


print("Tamaños")
print(len(full_dataset))
print(TRAIN_SIZE, VAL_SIZE, TEST_SIZE)



# del full dataset usa el random split para obtener los dos conjuntos de entrenamiento y validacion
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [TRAIN_SIZE, VAL_SIZE, TEST_SIZE])

print(len(train_dataset), len(val_dataset), len(test_dataset))

# ahora crea los data loader para entrenamiento y validacion
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

prueba_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# el dataloader ya nos permite que iteremos dentro del objeto
pcgs, mask= next(iter(train_loader))
print(pcgs.shape, mask.shape)

for i, (x, y) in enumerate(train_loader):
    print(i, x.shape, y.shape)
    if i==3: break



pcgs, mask = next(iter(train_loader))


def plot_mini_batch(pcgs, mask):
    plt.figure(figsize=(15, 15))
    for i in range(BATCH_SIZE):  # hasta 32
        plt.subplot(4, 2, i + 1)  # subplot nro filas, nro columnas, posicion. La primera imagen empieza en la pos=1
        pcgs = pcgs.squeeze(1)
        mask = mask.squeeze(1)

        pcg = pcgs.numpy()  


        plt.plot(pcg[i, :])
        plt.plot(mask[i, :])


        plt.axis('On')
    plt.tight_layout()
    plt.show()

 
plot_mini_batch(pcgs, mask)     # mostramos las seniales







def accuracy(model, loader):
    correct = 0  # acumulador de elementos correctos
    intersection = 0  # acumulador de elementos en la interseccion
    denom = 0
    union = 0
    total = 0
    cost = 0.
    model = model.to(device=device)  # ponemos el modelo en el dispositivo
    with torch.no_grad():  # no calculamos gradientes
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # mandamos elementos al dispositivo
            y = y.to(device=device, dtype=torch.long)  # mandamos elementos al dispositivo

            #y = torch.cat([y1, y2 ,y3], dim=1)           # concantena a lo largo de la dimension de los canales
            scores = model(x)
            
            y = y.squeeze(1)
            cost += F.cross_entropy(input=scores, target=y).item() # item nos devuielve el valor escalar del tensor
            #cost = F.mse_loss(input=scores, target=y)
           
            preds  = torch.argmax(scores, dim=1)
            
            correct += (preds == y).sum()
            
            total += torch.numel(preds)
            
            #dice coefficient
            #intersection += (preds*y).sum()                      # multiplicacion de la mascara con lo que nos da nuestro modelo                   
            intersection += (preds&y).sum()  
            
            denom += (preds + y).sum()
            
            dice = 2*intersection/(denom + 1e-8)
            
            #intersection over union
            #union += (preds + y - preds*y).sum()
            union += (preds | y).sum()

            iou =(intersection)/(union + 1e-8)
            

        return cost / len(loader), float(correct) / total, dice, iou





# funcion de entrenamiento.

def train(model, optimiser, scheduler=None, epochs= 50, store_every=5):
    model = model.to(device=device)                             # pasa el modelo a GPU
    train_acc_mb=[]
    train_cost_mb =[]
    iou_acc_mb = []
    dice_acc_mb=[]
    val_acc_mb =[]
    val_cost_mb=[]
    test_dice_mb = []
    test_iou_mb = []
    test_cost_mb = []
    test_acc_mb = []
    train_acc_epoch = []
    train_cost_epoch = []
    iou_acc_epoch = []
    dice_acc_epoch = []
    val_acc_epoch = []
    val_cost_epoch = []
    
    test_iou_epoch = []
    test_dice_epoch = []
    test_acc_epoch = []
    test_cost_epoch = []
    for epoch in range(epochs):
        train_correct_num = 0                                        # acumuladores que se resetean en cada epoch
        train_total = 0                                              # acumuladores que se resetean en cada epoch
        train_cost_acum = 0.                                         # acumuladores que se resetean en cada epoch
        val_cost_acum = 0
        val_acc_acum = 0
        dice_acum = 0
        iou_acum = 0
        test_acc_acum = 0
        test_cost_acum = 0
        test_iou_acum = 0
        test_dice_acum = 0

        for mb, (x, y) in enumerate(train_loader, start=1):          # iteracion para cada mini batch (mb)
            model.train()                                            # pone el modelo en modo entrenamiento 
            x = x.to(device=device, dtype=torch.float32)             # pone minibatches en el dispositivo
            y = y.to(device=device, dtype=torch.long)           # para poder calcular la funcion de costo, recordemos que y tiene dimensiones 32,1,224,224. Tiene 1 canal y si mandamos un canal en la funcion de costo no lo va a tomar. entonces lo sacacamos para que quede 32,224,224. Quitamos la dimension 1

            y = y.squeeze(1)
            
            scores = model(x)                                        # calcula scores
            cost = F.cross_entropy(input=scores, target=y)

            optimiser.zero_grad()                                    # resetea gradientes en caso que haya gradientes acumulados   
            cost.backward()
            optimiser.step()

            train_predictions = torch.argmax(scores, dim=1)  # scores nos devuelve 3 canales. Nos quedamos con el canal que tiene mayor probabilidad. dim=1 es la dimension de los canales
            
            train_correct_num += (train_predictions == y).sum()    
 
            train_total += torch.numel(train_predictions)
            train_cost_acum += cost.item()                          # acumula el costo. con .item() extraemos el valor ya que es un tensor de pytorch

            #if mb%store_every == 0:
            val_cost, val_acc, val_dice, val_iou = accuracy(model, val_loader)       # calucla accuracy
            test_cost, test_acc, test_dice, test_iou = accuracy(model, test_loader)       # calucla accuracy

            val_cost_acum += val_cost
            val_acc_acum += val_acc
            dice_acum += val_dice
            iou_acum += val_iou
            test_dice_acum += test_dice
            test_iou_acum += test_iou
            test_acc_acum += test_acc
            test_cost_acum += test_cost

            train_acc_mb_float = float(train_correct_num)/train_total        # cada 25 iteraciones calcula train_acc. Correctos sobre el total
            train_cost_float = float(train_cost_acum)/mb                     # es train_cost_acum normalizado

            val_cost_mb_float = float(val_cost_acum)/mb
            val_acc_mb_float = float(val_acc_acum)/mb
            dice_acc_mb_float = float(dice_acum)/mb
            iou_acc_mb_float = float(iou_acum)/mb
            test_dice_mb_float = float(test_dice_acum)/mb
            test_iou_mb_float = float(test_iou_acum)/mb
            test_cost_mb_float = float(test_cost_acum)/mb
            test_acc_mb_float = float(test_acc_acum)/mb

            print(f'epoch: {epoch}, mb: {mb}, train cost: {train_cost_float:.3f}, val cost: {val_cost:.3f},'
                  f'train acc: {train_acc_mb_float:.3f}, val acc: {val_acc:.3f},'
                  f'dice: {val_dice:.3f}, iou: {val_iou:.3f}')

            # Save data
            train_acc_mb.append(train_acc_mb_float)
            train_cost_mb.append(train_cost_float)
            val_acc_mb.append(val_acc)
            val_cost_mb.append(val_cost)
            dice_acc_mb.append(val_dice.item())
            iou_acc_mb.append(val_iou.item())


            
            test_dice_mb.append(test_dice_mb_float)
            test_iou_mb.append(test_iou_mb_float)
            test_cost_mb.append(test_cost_mb_float)
            test_acc_mb.append(test_acc_mb_float)


        train_acc_epoch.append(train_acc_mb_float) 
        train_cost_epoch.append(train_cost_float) 
        val_acc_epoch.append(val_acc_mb_float) 
        val_cost_epoch.append(val_cost_mb_float) 
        dice_acc_epoch.append(dice_acc_mb_float) 
        iou_acc_epoch.append(iou_acc_mb_float) 


        
        test_acc_epoch.append(test_acc_mb_float) 
        test_cost_epoch.append(test_cost_mb_float) 
        test_dice_epoch.append(test_dice_mb_float) 
        test_iou_epoch.append(test_iou_mb_float)


    return train_acc_mb ,train_cost_mb,val_acc_mb,val_cost_mb, dice_acc_mb, iou_acc_mb, \
            test_acc_mb, test_cost_mb, test_dice_mb, test_iou_mb, train_acc_epoch, train_cost_epoch, \
            val_acc_epoch, val_cost_epoch, dice_acc_epoch, iou_acc_epoch, test_acc_epoch, test_cost_epoch,\
            test_dice_epoch, test_iou_epoch


# bloque con constantes, ya que usa kernel_size=3  , stride=1 y poadding=1 en todos los casos
# padding =same es para que queden las señales de salida del mismo tamaño que las de entrada
# ahora cada vez que se use un bloque de convolucion solo se le pasa cantidad de canales de entrada y cantidad de canales de salida
# lo demas ya queda implicito
class Conv_3_k(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=1, padding='same')  # de movida va conv 1d
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
            #nn.MaxPool1d(2, 2),
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
        self.first_conv = Double_Conv(channels_in, channels)       # 64, 1024
        self.down_conv1 = Down_Conv(channels, 2 * channels)        # 128, 512
        self.down_conv2 = Down_Conv(2 * channels, 4 * channels)    # 256, 256
        self.down_conv3 = Down_Conv(4 * channels, 8 * channels)    # 512, 128

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



# Prueba de dimensiones

def test():
    x = torch.randn((8, 1, 1024))
    model = UNET(1, 64, 3)
    return model(x)


preds = test()
print(preds.shape)

# comienza el entrenamiento

model = UNET(1, 64, 3)
#preds = test()




#optimiser_unet = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
optimiser_unet = torch.optim.Adam(model.parameters(),lr=0.001)
#epochs = 5

train_acc_mb ,train_cost_mb,val_acc_mb,val_cost_mb, dice_acc_mb, iou_acc_mb, test_acc_mb, test_cost_mb, test_dice_mb, test_iou_mb, train_acc_epoch, train_cost_epoch, val_acc_epoch, val_cost_epoch, dice_acc_epoch, iou_acc_epoch, test_acc_epoch, test_cost_epoch, test_dice_epoch, test_iou_epoch  = train(model, optimiser_unet)

#plt.figure(figsize=(20,20))
plt.figure()
plt.title('Accuracy')
plt.xlabel('Mini Batch')
plt.plot(train_acc_mb,label='Train')
plt.plot(val_acc_mb,label='Val')
plt.plot(test_acc_mb,label='Val')
plt.legend()
plt.savefig('Accuracy_MiniBatch.png', dpi=300)



#plt.figure(figsize=(15,15))
plt.figure()
plt.title('IOU')
plt.xlabel('Mini Batch')
plt.plot(iou_acc_mb,label='IOU')
plt.plot(test_iou_mb,label='test IOU')
plt.plot(dice_acc_mb,label='Dice')
plt.plot(test_dice_mb,label='test Dice')
plt.legend()
plt.savefig('IOU_MiniBatch.png', dpi=300)



#plt.figure(figsize=(15,15))
plt.figure()
plt.title('Cost')
plt.xlabel('Mini Batch')
plt.plot(train_cost_mb,label='Train')
plt.plot(val_cost_mb,label='Val')
plt.plot(test_cost_mb,label='test')
plt.legend()
plt.savefig('Cost_MiniBatch.png', dpi=300)



#plt.figure(figsize=(15,15))
plt.figure()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.plot(train_acc_epoch,label='Train')
plt.plot(val_acc_epoch,label='Val')
plt.plot(test_acc_epoch,label='test')
plt.legend()
plt.savefig('Accuracy_Epoch.png', dpi=300)



#plt.figure(figsize=(15,15))
plt.figure()
plt.title('IOU')
plt.xlabel('Epoch')
plt.plot(iou_acc_epoch,label='IOU')
plt.plot(test_iou_epoch,label='test IOU')
plt.plot(dice_acc_epoch,label='Dice')
plt.plot(test_dice_epoch,label='test Dice')
plt.legend()
plt.savefig('IOU_Epoch.png', dpi=300)


#plt.figure(figsize=(15,15))
plt.figure()
plt.title('Cost')
plt.xlabel('Epoch')
plt.plot(train_cost_epoch,label='Train')
plt.plot(val_cost_epoch,label='Val')
plt.plot(test_cost_epoch,label='test')
plt.legend()
plt.savefig('Cost_Epoch.png', dpi=300)



# Guardo el modelo
model_path = './model.pth'
torch.save(model.state_dict(),model_path)
#print(model.state_dict())


# Cargo el modelo previamente guardado

modelo = UNET(1,64,3)
modelo.load_state_dict(torch.load(model_path))
modelo.eval()


# Pruebas de validacion del modelo entrenado

pcgs_val, mask_val = next(iter(val_loader))
#print(pcgs.shape, mask_val.shape)

pcgs_val = pcgs_val.to(device, dtype=torch.float32)    # pongo seniales en gpu
modelo = modelo.to(device)                             # mando el modelo a GPU
with torch.no_grad():
    scores = modelo(pcgs_val)                          # calculo scores (tienen dos canales con la pribabilidad que pertenezca a una u otra clase)
    preds = torch.argmax(scores, dim=1).float()        # calcula predicciones (como se queda con el mayor en la dimension de canales, da la mascara)

pcgs_val = pcgs_val.squeeze(1).cpu().numpy() 
preds = preds.cpu().numpy()                        #
   
mask_val = mask_val.squeeze(1).cpu().numpy()

print(pcgs_val.shape,preds.shape,mask_val.shape)

 
# def plot_mini_batch_salida(pcgs, mask, preds):
#     plt.figure(figsize=(15,15))
#     for i in range(BATCH_SIZE):              #hasta 8        
#         plt.subplot(4, 2,i+1)               # subplot nro filas, nro columnas, posicion. La primera imagen empieza en la pos=1 
        
       
#         plt.plot(pcgs[i,:], label='pcg')     
#         plt.plot(mask[i,:], label='mask') 
#         plt.plot(preds[i,:], label='pred')
  
        
#         plt.axis('On')
#         plt.legend()
#     plt.show()
   
def plot_mini_batch_salida(pcgs, mask, preds):
    
    mask_base= np.zeros(mask.shape)
    mask_base[mask == 0] = 1
    mask_S1= np.zeros(mask.shape)
    mask_S1[mask == 1] = 1
    mask_S2= np.zeros(mask.shape)
    mask_S2[mask == 2] = 1
    
    preds_base= np.zeros(preds.shape)
    preds_base[preds == 0] = 1
    preds_S1= np.zeros(preds.shape)
    preds_S1[preds == 1] = 1
    preds_S2= np.zeros(preds.shape)
    preds_S2[preds == 2] = 1
    
    plt.figure(figsize=(15,15))
    for i in range(BATCH_SIZE):              #hasta 8        
        plt.subplot(4, 2,i+1)               # subplot nro filas, nro columnas, posicion. La primera imagen empieza en la pos=1 
        
        
        plt.plot(pcgs[i,:], 'b',label='pcg')     
        plt.plot(mask_base[i,:], 'r',label='mask') 
        plt.plot(preds_base[i,:],'g', label='pred')
  
        
        plt.axis('On')
        plt.legend()
        plt.title('Linea de Base')
    plt.savefig('Linea_de_Base.png', dpi=1200)    
    
    plt.figure(figsize=(15,15))
    for i in range(BATCH_SIZE):              #hasta 8        
        plt.subplot(4, 2,i+1)               # subplot nro filas, nro columnas, posicion. La primera imagen empieza en la pos=1 
        
       
        plt.plot(pcgs[i,:], 'b',label='pcg')     
        plt.plot(mask_S1[i,:], 'r',label='mask') 
        plt.plot(preds_S1[i,:],'g', label='pred')
  
        
        plt.axis('On')
        plt.legend()
        plt.title('S1')
    plt.savefig('S1.png', dpi=1200)  
     
    plt.figure(figsize=(15,15))
    for i in range(BATCH_SIZE):              #hasta 8        
        plt.subplot(4, 2,i+1)               # subplot nro filas, nro columnas, posicion. La primera imagen empieza en la pos=1 
        
       
        plt.plot(pcgs[i,:], 'b',label='pcg')     
        plt.plot(mask_S2[i,:], 'r',label='mask') 
        plt.plot(preds_S2[i,:],'g', label='pred')
  
        
        plt.axis('On')
        plt.legend()
        plt.title('S2')
    plt.savefig('S2.png', dpi=1200) 
    plt.show()
    

plot_mini_batch_salida(pcgs_val, mask_val, preds)    



# for i in range(7):
#     plt.figure()  
#     plt.plot(pcgs_val[i,:], label='pcg')     
#     plt.plot(preds[i,:], label='pred')   
#     plt.plot(mask_val[i,:],label='mask') 
#     plt.legend()



pcgs_test, mask_test = next(iter(test_loader))

pcgs_test = pcgs_test.to(device, dtype=torch.float32)    # pongo seniales en gpu
modelo = modelo.to(device)                             # mando el modelo a GPU
with torch.no_grad():
    scores = modelo(pcgs_test)                          # calculo scores (tienen dos canales con la pribabilidad que pertenezca a una u otra clase)
    preds = torch.argmax(scores, dim=1).float()        # calcula predicciones (como se queda con el mayor en la dimension de canales, da la mascara)

pcgs_test = pcgs_test.squeeze(1).cpu().numpy() 
preds = preds.cpu().numpy()                        #   
mask_test = mask_test.squeeze(1).cpu().numpy()


print(pcgs_test.shape,preds.shape,mask_test.shape)


plot_mini_batch_salida(pcgs_test, mask_test, preds)  
test_cost, tes_acc, test_dice, test_iou = accuracy(model, test_loader)

print(test_cost, tes_acc, test_dice, test_iou)




pcgs_prueba, mask_prueba = next(iter(prueba_loader))
    
pcgs_prueba = pcgs_prueba.to(device, dtype=torch.float32)    # pongo seniales en gpu
modelo = modelo.to(device)                             # mando el modelo a GPU
with torch.no_grad():
    scores = modelo(pcgs_prueba)                          # calculo scores (tienen dos canales con la probabilidad que pertenezca a una u otra clase)
    preds = torch.argmax(scores, dim=1).float()        # calcula predicciones (como se queda con el mayor en la dimension de canales, da la mascara)

pcgs_prueba = pcgs_prueba.squeeze(1).cpu().numpy() 
preds = preds.cpu().numpy()                        #   
   
    