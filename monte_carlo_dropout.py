import torchxrayvision as xrv
import torch
import pandas as pd
import numpy as np 
import cv2
from tqdm import tqdm

INVENTARIO_CSV = 'inventario_base_dados.csv'

model = xrv.models.DenseNet(weights="densenet121-res224-all")

def ativar_dropout_densenet(model, drop_rate=0.2):
    model.eval()

    for modulo in model.modules():
        if modulo.__class__.__name__ == '_DenseLayer':
            modulo.drop_rate = drop_rate
            modulo.train()

    for modulo in model.modules():
        if isinstance(modulo, torch.nn.BatchNorm2d):
            modulo.eval()

# função pronta pra pre-processar no padrão do DenseNet
# Tamanho: 224 × 224 pixels
# Valores entre -1024 e 1024 
# Formato: tensor PyTorch com shape (1, 1, 224, 224) 

def preprocessar(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = (img * 2048 / 255) - 1024
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
    ativar_dropout_densenet(model)
    print("Dropout ativado!")

    inventario = pd.read_csv(INVENTARIO_CSV)

    registros = []

    for _, row in tqdm(inventario.iterrows(), total=len(inventario), desc="MC Dropout"):
        img = preprocessar(row['caminho'])

        resultados = mc_dropout(model, img)
        resultados_array = np.stack(resultados)
        #print(resultados[0].shape)
        #print(resultados_array)
        media = np.mean(resultados_array, axis = 0)
        variancia = np.var(resultados_array, axis = 0)
        incerteza_geral = np.mean(variancia)

        registros.append({
            'nome_arquivo': row['nome_arquivo'],
            'classe':       row['classe'],
            'incerteza':    incerteza_geral,
        })

    df_resultado = pd.DataFrame(registros)
    df_resultado.to_csv('incerteza_mcdropout.csv', index=False)
    print("Salvo: incerteza_mcdropout.csv")

if __name__ == "__main__":
    main()
