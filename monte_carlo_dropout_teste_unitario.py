import torchxrayvision as xrv
import torch
import pandas as pd
import numpy as np 
import cv2
from tqdm import tqdm

INVENTARIO_CSV = 'inventario_base_dados.csv'

model = xrv.models.DenseNet(weights="densenet121-res224-all")

def ativar_dropout(model, drop_rate=0.2):
    model.eval()

    for modulo in model.modules():
        if modulo.__class__.__name__ == '_DenseLayer':
            modulo.drop_rate = drop_rate
            modulo.train()

    for modulo in model.modules():
        if isinstance(modulo, torch.nn.BatchNorm2d):
            modulo.eval()
    print(f"Drop rate = {drop_rate}")

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
    ativar_dropout(model)
    print("Dropout ativado!")

'''    
    inventario = pd.read_csv(INVENTARIO_CSV)
    primeira_linha = inventario.iloc[0]
    img = preprocessar(primeira_linha['caminho'])
    print(img.shape)

    resultados = mc_dropout(model, img)
    resultados_array = np.stack(resultados)
    #print(len(resultados))
    #print(resultados[0].shape)
    #print(model.pathologies)
    #print(resultados_array)
    media = np.mean(resultados_array, axis = 0)
    variancia = np.var(resultados_array, axis = 0)
    incerteza_geral = np.mean(variancia)

    print("Média shape:", media.shape)
    print("Variância shape:", variancia.shape)
    print("Média:", media)
    print("Variância:", variancia)
    print(f"Incerteza geral da imagem: {incerteza_geral:.6f}") 
'''
inventario = pd.read_csv(INVENTARIO_CSV)

amostra = inventario.groupby('classe').sample(n=50, random_state=42).reset_index(drop=True)

print(f"Amostra: {len(amostra)} imagens")
print(amostra['classe'].value_counts().to_string())

registros = []

for _, row in tqdm(amostra.iterrows(), total=len(amostra), desc="MC Dropout"):
    img = preprocessar(row['caminho'])
    resultados = mc_dropout(model, img)
    resultados_array = np.stack(resultados)
    
    media = np.mean(resultados_array, axis=0)
    variancia = np.var(resultados_array, axis=0)
    incerteza_geral = np.mean(variancia)
    
    registros.append({
        'nome_arquivo': row['nome_arquivo'],
        'classe':       row['classe'],
        'incerteza':    incerteza_geral,
        'media_pneumonia': float(media[0, 8]),  # índice 8 = Pneumonia
    })

df = pd.DataFrame(registros)
df.to_csv('amostra_mcdropout.csv', index=False)

print("\nIncerteza média por classe:")
print(df.groupby('classe')['incerteza'].mean().to_string())

if __name__ == "__main__":
    main()
