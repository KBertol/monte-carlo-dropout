import torchxrayvision as xrv
import torch
import skimage.io
import pandas as pd
import numpy as np 
import skimage.transform

INVENTARIO_CSV = 'inventario_base_dados.csv'

model = xrv.models.DenseNet(weights="densenet121-res224-all")

def ativar_dropout(model):
    for modulo in model.modules():
        if isinstance(modulo, torch.nn.Dropout):
            modulo.train()

# função pronta pra pre-processar no padrão do DenseNet
# Tamanho: 224 × 224 pixels
# Valores entre -1024 e 1024 
# Formato: tensor PyTorch com shape (1, 1, 224, 224) 

def preprocessar(caminho):
    img = skimage.io.imread(caminho, as_gray=True)
    img = skimage.transform.resize(img, (224, 224), anti_aliasing=True)
    img = (img * 2048) - 1024
    img = img.astype(np.float32)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img

def mc_dropout(model, img, n_rodadas = 50):
    resultados = []

    with torch.no_grad():
        for i in range(n_rodadas):
            saida = model(img)
            # mudar o tensor pra arrray numpy pra ficar mais fácil
            resultados.append(saida.numpy())
            print(f"Rodada {i+1}/{n_rodadas}", end="\r")
    return resultados

def main():
    model.eval()
    ativar_dropout(model)
    print("Dropout ativado!")

    inventario = pd.read_csv(INVENTARIO_CSV)
    primeira_linha = inventario.iloc[0]
    img = preprocessar(primeira_linha['caminho'])
    print(img.shape)

    resultados = mc_dropout(model, img)
    print(len(resultados))
    print(resultados[0].shape)
    print(model.pathologies)

if __name__ == "__main__":
    main()
