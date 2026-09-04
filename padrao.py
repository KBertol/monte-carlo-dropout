import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# 0. REPRODUTIBILIDADE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. CONFIGURAÇÕES
BASE_PATH    = 'Base'
OUTPUT_CSV   = 'inventario_base_dados.csv'

PASTAS = {
    'valid_normais_frontal':      'Saudavel',
    'valid_Doentes_frontal':      'Doente',
    'valid_desconhecidos_frontal': 'Incerto',
}

EXTENSOES_VALIDAS = ('.png', '.jpg', '.jpeg', '.dcm')

# 2. MAPEAMENTO
print("=" * 55)
print("  INVENTÁRIO DA BASE DE DADOS")
print("=" * 55)

dados = []

for pasta_nome, classe in PASTAS.items():
    caminho_completo = os.path.join(BASE_PATH, pasta_nome)

    if not os.path.exists(caminho_completo):
        print(f"\n[AVISO] Pasta '{pasta_nome}' não encontrada em '{caminho_completo}'!")
        continue

    arquivos = os.listdir(caminho_completo)

    for arquivo in tqdm(arquivos, desc=f"Mapeando {classe}"):
        if arquivo.lower().endswith(EXTENSOES_VALIDAS):
            dados.append({
                'caminho':      os.path.join(caminho_completo, arquivo),
                'nome_arquivo': arquivo,
                'classe':       classe,
            })

# 3. SALVAR CSV
df = pd.DataFrame(dados)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 55)
print(f"  Inventário salvo em: {OUTPUT_CSV}")
print(f"  Total de imagens   : {len(df)}")
print("=" * 55)

if not df.empty:
    print("\nDistribuição por classe:")
    print(df['classe'].value_counts().to_string())
else:
    print("\n[ERRO] Nenhuma imagem encontrada. Verifique o nome da pasta Base.")
