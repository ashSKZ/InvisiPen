import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modelo import LetrasCNN

#para utilizar con GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Entrenando en :", device)

#el preprocesamiento de los datos de entrenamiento, EMNIST tiene una rotaci'on de 90 grados y un volteo horizontal, por lo que se corrige con una rotación de -90 grados y un volteo horizontal
transformaciones = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.transpose(1,2)),
    transforms.Normalize((0.5,), (0.5,))
])

#debido a que EMINIST utiliza las etiquetas de 1 a 26, se ajusta el rango
correccion_etiquetas = lambda y: y - 1

#descargar el dataset
train_dataset=datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transformaciones, target_transform=correccion_etiquetas)
test_dataset=datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transformaciones, target_transform=correccion_etiquetas)
train_loader=DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=128, shuffle=False)

modelo = LetrasCNN(num_clases=26).to(device)
criterio=nn.CrossEntropyLoss()
optimizador=optim.Adam(modelo.parameters(), lr=0.001)

epocas = 5
for epoca in range(epocas):
    modelo.train()
    perdida_total = 0.0
    correctos = 0
    total = 0
    for lotes, (imagenes, etiquetas) in enumerate(train_loader):
        imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
        predicciones = modelo(imagenes)
        loss = criterio(predicciones, etiquetas)
        optimizador.zero_grad()
        loss.backward()
        optimizador.step()
        perdida_total += loss.item()
        _, predicciones_clase = torch.max(predicciones, 1)
        correctos += (predicciones_clase == etiquetas).sum().item()
        total += etiquetas.size(0)
        if (lotes + 1) % 200 == 0:
            print(f"Época [{epoca + 1}/{epocas}], Lote [{lotes + 1}/{len(train_loader)}], Pérdida: {perdida_total / (lotes + 1):.4f}, Precisión: {100 * correctos / total:.2f}%")
    precision = 100 * correctos / total
    print(f"Época [{epoca + 1}/{epocas}] finalizada. Pérdida promedio: {perdida_total / len(train_loader):.4f}, Precisión: {precision:.2f}%")

#evalucion del modelo en el conjunto de prueba
modelo.eval()
correctos_test = 0
total_test = 0

with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
        predicciones = modelo(imagenes)
        _, predicciones_clase = torch.max(predicciones, 1)
        correctos_test += (predicciones_clase == etiquetas).sum().item()
        total_test += etiquetas.size(0)

precision_test = 100 * correctos_test / total_test
print(f"Precisión en el conjunto de prueba: {precision_test:.2f}%")

torch.save(modelo.state_dict(), "modelo_letras.pth")
print("Modelo guardado como modelo_letras.pth")
