import torch
import torch.nn as nn

class LetrasCNN(nn.Module):
    def __init__(self,num_clases=26):
        super(LetrasCNN, self).__init__()

        #bloque convolucional 1 (1,28,28)->(16,14,14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #bloque convolucional 2 (16,14,14)->(32,7,7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc=nn.Sequential(
            nn.Dropout(p=0.5), #para evitar el overfitting por los trazos manuakes
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, num_clases) 
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  #aAplanar la salida para la capa totalmente conectada
        x = self.fc(x)
        return x

if __name__ == "__main__":
    modelo=LetrasCNN(num_clases=26)  #26 letraas
    tensor_prueba=torch.randn(1,1,28,28)  #tamaño de entrada (batch_size, canales, alto, ancho)
    salida=modelo(tensor_prueba)
    print("Tamaño de salida:", salida.shape)  