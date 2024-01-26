
import pandas as pd #biblioteca para manupulacao de dados
from unidecode import unidecode #edicao de caracteres
import matplotlib.pyplot as plt #plotagem de graficos
import seaborn as sns #biblioteca com base no matplotlib com melhor design para plotagem
import numpy as np #operacoes com array, validacoes cruzadas

from sklearn.metrics import mean_squared_error, mean_absolute_error # Aplicação de metricas Mean Squared Error (MSE) e Mean Absolute Error (MAE)

'''Dados de Julho de 2023'''

#Ler arquivo .csv, gerando datafreame
df23 = pd.read_csv("d_frota_por_uf_municipio_combustivel_julho_2023.csv",
                         encoding = "UTF-8", sep = ";", usecols = ["UF",'Municipio',"Combustivel Veiculo","Qtd. Veiculos"])

print(df23.info()) #Qtd de entradas de dados = 47601
print(df23.head()) #Mostrar 5 primeiras linhas da tabela

#Substituir espacos (onde houver) dos titulos das colunas por "_"
df23.columns = df23.columns.str.replace(' ', '_')
print(df23.head())

print(df23['UF'].unique())
#Remover acentos e trocar espacos por "_" das linhas da coluna 'UF'
df23['UF'] = df23['UF'].apply(unidecode).str.replace(' ', '_')
#Deletar dados da coluna 'UF' sem identificacao de estados
df23 = df23[~df23['UF'].isin(['Nao_Identificado', 'Nao_se_Aplica', 'Sem_Informacao'])]
print(df23['UF'].unique())

print(df23['Combustivel_Veiculo'].unique())
#Remover acentos e trocar espacos por "_" das linhas da coluna 'Combustivel_Veiculo'
df23['Combustivel_Veiculo'] = df23['Combustivel_Veiculo'].apply(unidecode).str.replace(' ', '_')
#Deletar dados da coluna 'Combustivel_Veiculo' sem identificacao de tipo de combustivel
df23 = df23[~df23['Combustivel_Veiculo'].isin(['Nao_Identificado', 'Sem_Informacao', 'VIDE/CAMPO/OBSERVACAO'])]
print(df23['Combustivel_Veiculo'].unique())

print(df23['Municipio'].unique())
#Remover acentos e trocar espacos por "_" das linhas da coluna 'Municipio'
df23['Municipio'] = df23['Municipio'].apply(unidecode).str.replace(' ', '_')
print(df23['Municipio'].unique())

#Reagrupamento de veiculos Hibridos Completos (HEV) e Hibridos Leves (MHEV), com excessao dos Hibridos Plugin (PHEV): HIBRIDO
#Veiculos Flex, Gasolina, Alcool, GNV, Diesel (nao eletrificados), serao reagrupados como "COMBUSTAO"
#Veiculos 100% eletricos (BEV) adicionados ao novo agrupamento: ELETRICO
HIBRIDO = ['GASOLINA/ALCOOL/ELETRICO', 'GASOLINA/ELETRICO', 'DIESEL/ELETRICO', 'HIBRIDO', 'ETANOL/ELETRICO']
COMBUSTAO = ['ALCOOL', 'ALCOOL/GASOLINA', 'DIESEL', 'GASOLINA', 'GASOLINA/ALCOOL/GAS_NATURAL', 'GASOL/GAS_NATURAL_COMBUSTIVEL', 'GASOLINA/GAS_NATURAL_VEICULAR', 'ALCOOL/GAS_NATURAL_COMBUSTIVEL', 'ALCOOL/GAS_NATURAL_VEICULAR', 'GAS_METANO', 'GAS_NATURAL_VEICULAR', 'GASOGENIO', 'DIESEL/GAS_NATURAL_VEICULAR', 'DIESEL/GAS_NATURAL_COMBUSTIVEL', 'GAS/NATURAL/LIQUEFEITO']
ELETRICO = ['ELETRICO/FONTE_EXTERNA', 'ELETRICO/FONTE_INTERNA', 'ELETRICO']
df23['Combustivel_Veiculo'] = df23['Combustivel_Veiculo'].apply(lambda x: 'HIBRIDO' if x in HIBRIDO else x)
df23['Combustivel_Veiculo'] = df23['Combustivel_Veiculo'].apply(lambda x: 'COMBUSTAO' if x in COMBUSTAO else x)
df23['Combustivel_Veiculo'] = df23['Combustivel_Veiculo'].apply(lambda x: 'ELETRICO' if x in ELETRICO else x)
print(df23['Combustivel_Veiculo'].unique())
# Para melhorar a visualizacao e evitar a repeticao dos nomes dos combustiveis, o agrupamento sera feito atraves do comando groupby() com somatoria dos valores
df23 = df23.groupby(['UF', 'Municipio', 'Combustivel_Veiculo']).sum().reset_index()

#df23.to_csv('df23_tratado.csv', index=False) #criar novo arquivo do dataframe
print(df23) 

''' Dados de Julho de 2022'''

df22 = pd.read_csv("d_frota_por_uf_municipio_combustivel_julho_2022.csv",
                         encoding = "UTF-8", sep = ";")

print(df22.info()) #Qtd de entradas de dados 

# Substituir espacos (onde houver) dos titulos das colunas por "_"
df22.columns = df22.columns.str.replace(' ', '_')
# Tratamento das colunas: remocao de acentos, troca de espacos por "_" e filtragem de dados sem info
df22['UF'] = df22['UF'].apply(unidecode).str.replace(' ', '_')
df22 = df22[~df22['UF'].isin(['Nao_Identificado', 'Nao_se_Aplica', 'Sem_Informacao'])]

df22['Combustivel_Veiculo'] = df22['Combustivel_Veiculo'].apply(unidecode).str.replace(' ', '_')
df22 = df22[~df22['Combustivel_Veiculo'].isin(['Nao_Identificado', 'Sem_Informacao', 'VIDE/CAMPO/OBSERVACAO'])]

df22['Municipio'] = df22['Municipio'].apply(unidecode).str.replace(' ', '_')

# Reagrupamento de veiculos Hibridos(exceto PHEV), a Combustao e Eletricos
HIBRIDO = ['GASOLINA/ALCOOL/ELETRICO', 'GASOLINA/ELETRICO', 'DIESEL/ELETRICO', 'HIBRIDO', 'ETANOL/ELETRICO']
COMBUSTAO = ['ALCOOL', 'ALCOOL/GASOLINA', 'DIESEL', 'GASOLINA', 'GASOLINA/ALCOOL/GAS_NATURAL', 'GASOL/GAS_NATURAL_COMBUSTIVEL', 'GASOLINA/GAS_NATURAL_VEICULAR', 'ALCOOL/GAS_NATURAL_COMBUSTIVEL', 'ALCOOL/GAS_NATURAL_VEICULAR', 'GAS_METANO', 'GAS_NATURAL_VEICULAR', 'GASOGENIO', 'DIESEL/GAS_NATURAL_VEICULAR', 'DIESEL/GAS_NATURAL_COMBUSTIVEL', 'GAS/NATURAL/LIQUEFEITO']
ELETRICO = ['ELETRICO/FONTE_EXTERNA', 'ELETRICO/FONTE_INTERNA', 'ELETRICO']
df22['Combustivel_Veiculo'] = df22['Combustivel_Veiculo'].apply(lambda x: 'HIBRIDO' if x in HIBRIDO else x)
df22['Combustivel_Veiculo'] = df22['Combustivel_Veiculo'].apply(lambda x: 'COMBUSTAO' if x in COMBUSTAO else x)
df22['Combustivel_Veiculo'] = df22['Combustivel_Veiculo'].apply(lambda x: 'ELETRICO' if x in ELETRICO else x)
print(df22['Combustivel_Veiculo'].unique())
# Para melhorar a visualizacao e evitar a repeticao dos nomes dos combustiveis, o agrupamento sera feito atraves do comando groupby() com somatoria dos valores
df22 = df22.groupby(['UF', 'Municipio', 'Combustivel_Veiculo']).sum().reset_index()

print(df22) 

''' Dados de Julho de 2021'''

# Carregar os dados
df21 = pd.read_csv("d_frota_por_uf_municipio_combustivel_julho_2021.csv",
                   encoding="UTF-8", sep=";")

print(df21.info()) #Qtd de entradas de dados  

# Substituir espacos (onde houver) dos titulos das colunas por "_"
df21.columns = df21.columns.str.replace(' ', '_')
# Tratamento das colunas: remocao de acentos, troca de espacos por "_" e filtragem de dados sem info
df21['UF'] = df21['UF'].apply(unidecode).str.replace(' ', '_')
df21 = df21[~df21['UF'].isin(['Nao_Identificado', 'Nao_se_Aplica', 'Sem_Informacao'])]

df21['Combustivel_Veiculo'] = df21['Combustivel_Veiculo'].apply(unidecode).str.replace(' ', '_')
df21 = df21[~df21['Combustivel_Veiculo'].isin(['Nao_Identificado', 'Sem_Informacao', 'VIDE/CAMPO/OBSERVACAO'])]

df21['Municipio'] = df21['Municipio'].apply(unidecode).str.replace(' ', '_')

# Reagrupamento de veiculos Hibridos, a Combustao e Eletricos
HIBRIDO = ['GASOLINA/ALCOOL/ELETRICO', 'GASOLINA/ELETRICO', 'DIESEL/ELETRICO', 'HIBRIDO', 'ETANOL/ELETRICO']
COMBUSTAO = ['ALCOOL', 'ALCOOL/GASOLINA', 'DIESEL', 'GASOLINA', 'GASOLINA/ALCOOL/GAS_NATURAL', 'GASOL/GAS_NATURAL_COMBUSTIVEL', 'GASOLINA/GAS_NATURAL_VEICULAR', 'ALCOOL/GAS_NATURAL_COMBUSTIVEL', 'ALCOOL/GAS_NATURAL_VEICULAR', 'GAS_METANO', 'GAS_NATURAL_VEICULAR', 'GASOGENIO', 'DIESEL/GAS_NATURAL_VEICULAR', 'DIESEL/GAS_NATURAL_COMBUSTIVEL', 'GAS/NATURAL/LIQUEFEITO']
ELETRICO = ['ELETRICO/FONTE_EXTERNA', 'ELETRICO/FONTE_INTERNA', 'ELETRICO']
df21['Combustivel_Veiculo'] = df21['Combustivel_Veiculo'].apply(lambda x: 'HIBRIDO' if x in HIBRIDO else x)
df21['Combustivel_Veiculo'] = df21['Combustivel_Veiculo'].apply(lambda x: 'COMBUSTAO' if x in COMBUSTAO else x)
df21['Combustivel_Veiculo'] = df21['Combustivel_Veiculo'].apply(lambda x: 'ELETRICO' if x in ELETRICO else x)
print(df21['Combustivel_Veiculo'].unique())

# Para melhorar a visualizacao e evitar a repeticao dos nomes dos combustiveis, o agrupamento sera feito atraves do comando groupby() com somatoria dos valores
df21 = df21.groupby(['UF', 'Municipio', 'Combustivel_Veiculo']).sum().reset_index()

print(df21)

'''Dados de Julho de 2020'''

df20 = pd.read_csv("d_frota_por_uf_municipio_combustivel_julho_2020.csv",
                   encoding="UTF-8", sep=";")

print(df20.info()) #Qtd de entradas de dados 

# Substituir espacos dos titulos das colunas por "_"
df20.columns = df20.columns.str.replace(' ', '_')
# Tratamento das colunas: remocao de acentos, troca de espacos por "_" e filtragem de dados sem info
df20[['UF', 'Municipio', 'Combustivel_Veiculo']] = df20[['UF', 'Municipio', 'Combustivel_Veiculo']].applymap(unidecode).applymap(lambda x: x.replace(' ', '_'))
df20 = df20[~df20['UF'].isin(['Nao_Identificado', 'Nao_se_Aplica', 'Sem_Informacao'])]
df20 = df20[~df20['Combustivel_Veiculo'].isin(['Nao_Identificado', 'Sem_Informacao', 'VIDE/CAMPO/OBSERVACAO'])]

# Reagrupamento de veiculos Hibridos, a Combustao e Eletricos
df20['Combustivel_Veiculo'] = df20['Combustivel_Veiculo'].apply(lambda x: 'HIBRIDO' if x in HIBRIDO else x)
df20['Combustivel_Veiculo'] = df20['Combustivel_Veiculo'].apply(lambda x: 'COMBUSTAO' if x in COMBUSTAO else x)
df20['Combustivel_Veiculo'] = df20['Combustivel_Veiculo'].apply(lambda x: 'ELETRICO' if x in ELETRICO else x)
print(df20['Combustivel_Veiculo'].unique())

# Agrupamento atraves do comando groupby() com somatoria dos valores
df20 = df20.groupby(['UF', 'Municipio', 'Combustivel_Veiculo']).sum().reset_index()

print(df20)

'''Dados de Julho de 2019'''

df19 = pd.read_csv("d_frota_por_uf_municipio_combustivel_julho_2019.csv",
                   encoding="UTF-8", sep=";")

print(df19.info()) #Qtd de entradas de dados 

# Substituir espacos dos titulos das colunas por "_"
df19.columns = df19.columns.str.replace(' ', '_')
# Tratamento das colunas: remocao de acentos, troca de espacos por "_" e filtragem de dados sem info
df19[['UF', 'Municipio', 'Combustivel_Veiculo']] = df19[['UF', 'Municipio', 'Combustivel_Veiculo']].applymap(unidecode).applymap(lambda x: x.replace(' ', '_'))
df19 = df19[~df19['UF'].isin(['Nao_Identificado', 'Nao_se_Aplica', 'Sem_Informacao'])]
df19 = df19[~df19['Combustivel_Veiculo'].isin(['VIDE/CAMPO/OBSERVACAO', 'NAPSo_se_Aplica', 'NAPSo_Identificado', 'Sem_InformaASSAPSo'])]

# Reagrupamento de veiculos Hibridos, a Combustao e Eletricos
HIBRIDO = ['GASOLINA/ALCOOL/ELETRICO', 'GASOLINA/ELETRICO', 'DIESEL/ELETRICO', 'HIBRIDO', 'ETANOL/ELETRICO']
COMBUSTAO = ['ALCOOL', 'ALCOOL/GASOLINA', 'DIESEL', 'GASOLINA', 'GASOLINA/ALCOOL/GAS_NATURAL', 'GASOL/GAS_NATURAL_COMBUSTIVEL', 'GASOLINA/GAS_NATURAL_VEICULAR', 'ALCOOL/GAS_NATURAL_COMBUSTIVEL', 'ALCOOL/GAS_NATURAL_VEICULAR', 'GAS_METANO', 'GAS_NATURAL_VEICULAR', 'GASOGENIO', 'DIESEL/GAS_NATURAL_VEICULAR', 'DIESEL/GAS_NATURAL_COMBUSTIVEL', 'GAS/NATURAL/LIQUEFEITO']
ELETRICO = ['ELETRICO/FONTE_EXTERNA', 'ELETRICO/FONTE_INTERNA', 'ELETRICO']

df19['Combustivel_Veiculo'] = df19['Combustivel_Veiculo'].apply(lambda x: 'HIBRIDO' if x in HIBRIDO else x)
df19['Combustivel_Veiculo'] = df19['Combustivel_Veiculo'].apply(lambda x: 'COMBUSTAO' if x in COMBUSTAO else x)
df19['Combustivel_Veiculo'] = df19['Combustivel_Veiculo'].apply(lambda x: 'ELETRICO' if x in ELETRICO else x)
print(df19['Combustivel_Veiculo'].unique())

# Agrupamento atraves do comando groupby() com somatoria dos valores
df19 = df19.groupby(['UF', 'Municipio', 'Combustivel_Veiculo']).sum().reset_index()

print(df19)

''' Dados de Julho de 2018'''

# Carregar os dados
df18 = pd.read_csv("d_frota_por_uf_municipio_combustivel_julho_2018.csv",
                   encoding="UTF-8", sep=";")

print(df18.info()) #Qtd de entradas de dados = 

# Substituir espacos (onde houver) dos titulos das colunas por "_"
df18.columns = df18.columns.str.replace(' ', '_')

# Tratamento das colunas: remocao de acentos, troca de espacos por "_" e filtragem de dados sem info
df18[['UF', 'Municipio', 'Combustivel_Veiculo']] = df18[['UF', 'Municipio', 'Combustivel_Veiculo']].applymap(unidecode).applymap(lambda x: x.replace(' ', '_'))
df18 = df18[~df18['UF'].isin(['Nao_Identificado', 'Nao_se_Aplica', 'Sem_Informacao'])]
df18 = df18[~df18['Combustivel_Veiculo'].isin(['VIDE/CAMPO/OBSERVACAO', 'NAPSo_se_Aplica', 'NAPSo_Identificado', 'Sem_InformaASSAPSo'])]

# Reagrupamento de veiculos Hibridos(exceto PHEV), a Combustao e Eletricos
HIBRIDO = ['GASOLINA/ALCOOL/ELETRICO', 'GASOLINA/ELETRICO', 'DIESEL/ELETRICO', 'HIBRIDO', 'ETANOL/ELETRICO']
COMBUSTAO = ['ALCOOL', 'ALCOOL/GASOLINA', 'DIESEL', 'GASOLINA', 'GASOLINA/ALCOOL/GAS_NATURAL', 'GASOL/GAS_NATURAL_COMBUSTIVEL', 'GASOLINA/GAS_NATURAL_VEICULAR', 'ALCOOL/GAS_NATURAL_COMBUSTIVEL', 'ALCOOL/GAS_NATURAL_VEICULAR', 'GAS_METANO', 'GAS_NATURAL_VEICULAR', 'GASOGENIO', 'DIESEL/GAS_NATURAL_VEICULAR', 'DIESEL/GAS_NATURAL_COMBUSTIVEL', 'GAS/NATURAL/LIQUEFEITO']
ELETRICO = ['ELETRICO/FONTE_EXTERNA', 'ELETRICO/FONTE_INTERNA', 'ELETRICO']
df18['Combustivel_Veiculo'] = df18['Combustivel_Veiculo'].apply(lambda x: 'HIBRIDO' if x in HIBRIDO else x)
df18['Combustivel_Veiculo'] = df18['Combustivel_Veiculo'].apply(lambda x: 'COMBUSTAO' if x in COMBUSTAO else x)
df18['Combustivel_Veiculo'] = df18['Combustivel_Veiculo'].apply(lambda x: 'ELETRICO' if x in ELETRICO else x)
print(df18['Combustivel_Veiculo'].unique())

# Para melhorar a visualizacao e evitar a repeticao dos nomes dos combustiveis, o agrupamento sera feito atraves do comando groupby() com somatoria dos valores
df18 = df18.groupby(['UF', 'Municipio', 'Combustivel_Veiculo']).sum().reset_index()

print(df18) 

'''Analise e plotagem de graficos - Crescimento Anual de 2020 a 2023'''

#Atribuindo dado "Ano" a cada dataframe
df20['Ano'] = 2020
df21['Ano'] = 2021
df22['Ano'] = 2022
df23['Ano'] = 2023

# Combinando os dataframes
df_combinados = pd.concat([df20, df21, df22, df23])

# Agrupamento por ano, tipo de combustivel e soma da quantidade de veiculos 
dados_anuais = df_combinados.groupby(['Ano', 'Combustivel_Veiculo'])['Qtd._Veiculos'].sum().reset_index()

# Filtrando somente por Eletrificados
eletrificados_ano = dados_anuais[dados_anuais['Combustivel_Veiculo'].isin(['CELULA_COMBUSTIVEL','HIBRIDO', 'HIBRIDO_PLUG-IN', 'ELETRICO'])]

# Escolhendo o estilo/cor do grafico
sns.set(style="whitegrid")

# Criando grafico de linha
plt.figure(figsize=(10, 6))
sns.lineplot(data=eletrificados_ano, x='Ano', y='Qtd._Veiculos', hue='Combustivel_Veiculo', marker='o')

# Adicionando Titulos e legendas
plt.title("Crescimento de veiculos Eletrificados ao longo dos anos") #Titulo do grafico
plt.xlabel('Ano') #Nome do eixo X
plt.ylabel('Quantidade de Veiculos') #Nome do eixo Y 
plt.xticks(eletrificados_ano['Ano'].unique()) #Ticks no eixo 'Ano'
plt.legend(title='Tipo de Combustivel') #Legenda do grafico, representando o tipo de combustivel plotado pra cada linha
for x, y in zip(eletrificados_ano['Ano'], eletrificados_ano['Qtd._Veiculos']):
    plt.text(x, y, f'{int(y)}', verticalalignment='bottom', horizontalalignment='right') #Mostrar os valores de cada Ano x Qtd de Veiculos

# Mostrar grafico
plt.show()

'''Analise e plotagem de graficos - Total de Veiculos Eletrificados por Estado'''

# Filtrando veiculos eletrificados
eletrificados_2023 = df23[df23['Combustivel_Veiculo'].isin(['CELULA_COMBUSTIVEL','HIBRIDO', 'HIBRIDO_PLUG-IN', 'ELETRICO'])]

# Agregando por UF, alterando nomes para siglas dos estados para melhor visualizacao
eletrificados_estados = eletrificados_2023.groupby('UF')['Qtd._Veiculos'].sum().reset_index()
eletrificados_estados2 = eletrificados_estados.copy() # criando copia para alteracao de nomes na coluna UF
siglas_UF = {
    'ACRE': 'AC', 'ALAGOAS': 'AL', 'AMAPA': 'AP', 'AMAZONAS': 'AM', 'BAHIA': 'BA', 'CEARA': 'CE', 
    'DISTRITO_FEDERAL': 'DF', 'ESPIRITO_SANTO': 'ES', 'GOIAS': 'GO', 'MARANHAO': 'MA', 
    'MATO_GROSSO': 'MT', 'MATO_GROSSO_DO_SUL': 'MS', 'MINAS_GERAIS': 'MG', 'PARA': 'PA', 
    'PARAIBA': 'PB', 'PARANA': 'PR', 'PERNAMBUCO': 'PE', 'PIAUI': 'PI', 'RIO_DE_JANEIRO': 'RJ', 
    'RIO_GRANDE_DO_NORTE': 'RN', 'RIO_GRANDE_DO_SUL': 'RS', 'RONDONIA': 'RO', 'RORAIMA': 'RR', 
    'SANTA_CATARINA': 'SC', 'SAO_PAULO': 'SP', 'SERGIPE': 'SE', 'TOCANTINS': 'TO'
}  
# trocando nomes dos estados por siglas
eletrificados_estados2['UF'] = eletrificados_estados2['UF'].map(siglas_UF).fillna(eletrificados_estados2['UF']) #incluindo no novo DF

# Escolhendo estilo/cor do grafico com o seaborn
sns.set(style="whitegrid")

# Criando grafico de barras
plt.figure(figsize=(12, 6))  #Ajute de tamanho
ax = sns.barplot(data=eletrificados_estados2, x='UF', y='Qtd._Veiculos', palette="Pastel2") #ajuste de layout e cores, cor Pastel2 

# Adiconando titulos de legendas
plt.title('Numero de Eletrificados em Julho de 2023 por estado')
plt.xlabel('Estado')
plt.ylabel('Quantidade')
for p in ax.patches:
    height = p.get_height()
    plt.text(p.get_x() + p.get_width() / 2., height + 3, f'{int(height)}', ha='center', va='bottom') #Posicionando legenda em cima das barras do grafico

# Mostrar grafico
plt.show()

'''Analise e plotagem de graficos - Relacao entre os tipos veiculos eletrificados'''

# Criando dicionario para armazenar os dados
dadosEV = {'Ano': [], 'ELETRICO': [], 'HIBRIDO': [], 'HIBRIDO_PLUG_IN': [], 'CELULA_COMBUSTIVEL': []}

for Ano, df in zip(range(2018, 2024), [df18, df19, df20, df21, df22, df23]):
    # Calcular a quantidade de veiculos por tipo
    cont_eletrico = df[df['Combustivel_Veiculo'] == 'ELETRICO']['Qtd._Veiculos'].sum()
    cont_hibrido = df[df['Combustivel_Veiculo'] == 'HIBRIDO']['Qtd._Veiculos'].sum()
    cont_hibrido_plug_in = df[df['Combustivel_Veiculo'] == 'HIBRIDO_PLUG-IN']['Qtd._Veiculos'].sum()
    cont_celula = df[df['Combustivel_Veiculo'] == 'CELULA_COMBUSTIVEL']['Qtd._Veiculos'].sum()

    # Armazenar os dados
    dadosEV['Ano'].append(Ano)
    dadosEV['ELETRICO'].append(cont_eletrico)
    dadosEV['HIBRIDO'].append(cont_hibrido)
    dadosEV['HIBRIDO_PLUG_IN'].append(cont_hibrido_plug_in)
    dadosEV['CELULA_COMBUSTIVEL'].append(cont_celula)

# Converter para DataFrame
df_comparacao = pd.DataFrame(dadosEV)

# Calcular o total de veículos eletrificados por ano
df_comparacao['Total'] = df_comparacao[[ 'ELETRICO', 'HIBRIDO', 'HIBRIDO_PLUG_IN','CELULA_COMBUSTIVEL']].sum(axis=1)

# Calcular as proporções percentuais de cada tipo de veículo
df_comparacao['ELETRICO_Pct'] = (df_comparacao['ELETRICO'] / df_comparacao['Total']) * 100
df_comparacao['HIBRIDO_Pct'] = (df_comparacao['HIBRIDO'] / df_comparacao['Total']) * 100
df_comparacao['HIBRIDO_PLUG_IN_Pct'] = (df_comparacao['HIBRIDO_PLUG_IN'] / df_comparacao['Total']) * 100
df_comparacao['CELULA_COMBUSTIVEL_Pct'] = (df_comparacao['CELULA_COMBUSTIVEL'] / df_comparacao['Total']) * 100

# Plotando o gráfico de proporções
plt.figure(figsize=(12, 6))

# variavel para medir a posicao, pois sera um grafico de barras empilhado
bottom = [0] * len(df_comparacao) 
# definção da estetica do grafico

for col, color, label in zip(['CELULA_COMBUSTIVEL_Pct', 'HIBRIDO_Pct', 'HIBRIDO_PLUG_IN_Pct', 'ELETRICO_Pct'], 
                             plt.cm.Set1.colors, 
                             ['Célula de Combustível', 'Híbrido', 'Híbrido Plug-In', 'Elétrico']):
    plt.bar(df_comparacao['Ano'], df_comparacao[col], bottom=bottom, color=color, label=label)
    
    # Adicionar texto nas barras
    for i in range(len(df_comparacao)):
        pct = df_comparacao[col].iloc[i]
        if pct > 0:  # Apenas mostrar se a porcentagem for maior que 0
            plt.text(df_comparacao['Ano'].iloc[i], bottom[i] + pct/2, f'{pct:.1f}%', ha='center', va='center')
    
    # Atualizar a posição do bottom para a próxima série
    bottom = [u + v for u, v in zip(bottom, df_comparacao[col])]

# Adicionando legendas e títulos
plt.xlabel('Ano')
plt.ylabel('Proporção (%)')
plt.title('Proporção entre Veículos Eletrificados (2018-2023)')
plt.legend()

# Mostrar o gráfico
plt.show()

'''Analise e plotagem de graficos - 10 Municipios do Brasil com mais veiculos eletrificados em Julho de 2023'''

# Filtrar o DataFrame para incluir apenas veículos eletrificados
df_eletrificados = df23[df23['Combustivel_Veiculo'].isin(['HIBRIDO', 'HIBRIDO_PLUG_IN', 'ELETRICO','CELULA_COMBUSTIVEL'])]

# Agrupar por município e somar a quantidade de veículos, ordenar os municípios em ordem descrescente e selecionar os 10 primeiros
soma_veiculos_por_municipio = df_eletrificados.groupby('Municipio')['Qtd._Veiculos'].sum().sort_values(ascending=False).head(10)

# Converter serie "soma_veiculos_por_municipio" em um dataframe, e resetar índice para que Municipio se torne uma coluna
top_10_municipios = soma_veiculos_por_municipio.reset_index()

# Adicionando uma coluna 'ID', somente para usar no eixo x, já que os nomes dos municipios serao mostrados somente no quadro da legenda
top_10_municipios['ID'] = range(1, len(top_10_municipios) + 1)

# Selecao de tamanho e tipo de grafico com seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))
ax = sns.barplot(x='ID', y='Qtd._Veiculos', hue='Municipio', data=top_10_municipios) #plotando com quadro de legendas com os nomes dos municipios para melhor leitura

# Adicionando nomes das legendas
plt.title('Top 10 Municípios no Brasil com Maior Quantidade de Veículos Eletrificados em Julho de 2023')
plt.xlabel('Município')
plt.ylabel('Quantidade de Veículos')
plt.xticks([]) # Para fins esteticos do grafico, nomes dos munucipios foram ocultos no eixo x, devido ao tamanho longo do nome de alguns dificultar a visualizacao 
plt.legend(title='Município') # Adicionar quadro com legenda de nomes dos municipios 

# Adicionar as anotacoes das quantidades acima das barras
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')
    
plt.show() # Mostrar o grafico

'''Analise e plotagem de graficos - Mapa de Calor de veiculos eletrificados no Brasil em 2023'''

import geopandas as gpd #biblioteca geopandas para lidar com Shapefiles e plotar os mapas de calor do mapa do Brasil

# Filtrar para veículos eletrificados
veiculos_eletrificados = ['ELETRICO', 'HIBRIDO', 'HIBRIDO_PLUG_IN', 'CELULA_COMBUSTIVEL']
df_eletrificados = df23[df23['Combustivel_Veiculo'].isin(veiculos_eletrificados)]

# Agregar dados por estado
df_eletrificados_por_estado = df_eletrificados.groupby('UF')['Qtd._Veiculos'].sum().reset_index()

# Como o Shapefile está identificando os estados por sigle, os nomes dos estados serão mapeados para as siglas correspondentes
mapeamento_nomes_para_siglas = {
    'ACRE': 'AC', 'ALAGOAS': 'AL', 'AMAPA': 'AP', 'AMAZONAS': 'AM', 'BAHIA': 'BA', 
    'CEARA': 'CE', 'DISTRITO_FEDERAL': 'DF', 'ESPIRITO_SANTO': 'ES', 'GOIAS': 'GO', 
    'MARANHAO': 'MA', 'MATO_GROSSO': 'MT', 'MATO_GROSSO_DO_SUL': 'MS', 
    'MINAS_GERAIS': 'MG', 'PARA': 'PA', 'PARAIBA': 'PB', 'PARANA': 'PR', 
    'PERNAMBUCO': 'PE', 'PIAUI': 'PI', 'RIO_DE_JANEIRO': 'RJ', 
    'RIO_GRANDE_DO_NORTE': 'RN', 'RIO_GRANDE_DO_SUL': 'RS', 'RONDONIA': 'RO', 
    'RORAIMA': 'RR', 'SANTA_CATARINA': 'SC', 'SAO_PAULO': 'SP', 'SERGIPE': 'SE', 
    'TOCANTINS': 'TO'
}
df_eletrificados_por_estado['UF'] = df_eletrificados_por_estado['UF'].map(mapeamento_nomes_para_siglas)

# Carregar os dados geográficos dos estados brasileiros
gdf_estados = gpd.read_file('BR_UF_2021.shp')

# Realizar a junção dos dados geográficos com os dados de veículos eletrificados
gdf_estados_veiculos = gdf_estados.set_index('SIGLA').join(df_eletrificados_por_estado.set_index('UF'))

# Plotar o mapa de calor, escolhendo tamanho, gradiente de cores (viridis) e adicionando legendas
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
gdf_estados_veiculos.plot(column='Qtd._Veiculos', cmap='viridis', ax=ax, legend=True,
                          legend_kwds={'label': "Quantidade de Veículos Eletrificados em 2023 por Estado",
                                       'orientation': "horizontal"})
ax.set_xticks([]) # Como é um mapa de calor, ocultando valores de x do shapefile no plot
ax.set_yticks([]) # Ocultando valores de y do shapefile no plot
plt.title('Distribuição de Veículos Eletrificados por Estado no Brasil em Junho de 2023') # titulo do gráfico
plt.show() # Exibir

### SERIES TEMPORAIS - MODELO DE REGRESSÃO LINEAR E POLINOMIAL
### PREPARACAO DOS DADOS - PREVISAO DE FROTA DE ELETRIFICADOS PARA OS PROXIMOS ANOS

# Concatenando os DataFrames de cada ano em um único DataFrame
df_total = pd.concat([df18, df19, df20, df21, df22, df23])

# Filtrar para veículos elétricos e híbridos e agrupar por ano
df_filtrado = df_total[df_total['Combustivel_Veiculo'].isin(['HIBRIDO', 'HIBRIDO_PLUG_IN', 'ELETRICO', 'CELULA_COMBUSTIVEL'])]
df_anual = df_filtrado.groupby('Ano')['Qtd._Veiculos'].sum().reset_index()  # Resetando o índice

# Preparando os dados
X = df_anual['Ano'].values.reshape(-1, 1)
y = df_anual['Qtd._Veiculos'].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Modelo de Regressão Linear
modelo_linear = LinearRegression()
modelo_linear.fit(X, y)
y_pred_linear = modelo_linear.predict(X)

# Modelo de Regressão Polinomial
grau_polynomial = 2  
poly_features = PolynomialFeatures(degree=grau_polynomial)
X_poly = poly_features.fit_transform(X)
modelo_polynomial = LinearRegression()
modelo_polynomial.fit(X_poly, y)
y_pred_poly = modelo_polynomial.predict(X_poly)

# Anos para os quais as previsões serão feitas
anos_futuros = np.array([2024, 2025, 2026]).reshape(-1, 1)

# Fazendo previsões com o modelo linear
previsoes_linear = modelo_linear.predict(anos_futuros)

# Preparando os dados para o modelo polinomial
anos_futuros_poly = poly_features.transform(anos_futuros)

# Fazendo previsões com o modelo polinomial
previsoes_poly = modelo_polynomial.predict(anos_futuros_poly)

# Exibir previsões
print("Previsões de Regressão Linear para 2024, 2025 e 2026: ", previsoes_linear)
print("Previsões de Regressão Polinomial para 2024, 2025 e 2026: ", previsoes_poly)

# Plotando os resultados com previsões
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Dados Reais', marker='o')
plt.plot(anos_futuros, previsoes_linear, label='Previsões Lineares', color='brown', marker='o')
plt.title('Previsões de Veículos Eletrificados para 2024, 2025 e 2026 - Modelo de Regressão Linear')
plt.xlabel('Ano')
plt.ylabel('Quantidade de Veículos')
plt.xticks(np.append(X, anos_futuros), rotation=45)  # Definindo os xticks
plt.legend()
plt.show()

# Plotando os resultados com previsões
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Dados Reais', marker='o')
plt.plot(anos_futuros, previsoes_poly, label='Previsões Polinomiais', color='green', marker='o')
plt.title('Previsões de Veículos Eletrificados para 2024, 2025 e 2026 - Modelo Polinomial')
plt.xlabel('Ano')
plt.ylabel('Quantidade de Veículos')
plt.xticks(np.append(X, anos_futuros), rotation=45)  # Definindo os xticks
plt.legend()
plt.show()

from sklearn.model_selection import TimeSeriesSplit

# Definir o número de splits
tscv = TimeSeriesSplit(n_splits=3)

# Listas para armazenar as métricas de cada modelo
def mean_percentage_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mse_linear_list, mae_linear_list, mpae_linear_list = [], [], []
mse_poly_list, mae_poly_list, mpae_poly_list = [], [], []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Modelo de Regressão Linear
    modelo_linear = LinearRegression()
    modelo_linear.fit(X_train, y_train)
    y_pred_linear = modelo_linear.predict(X_test)
    mse_linear_list.append(mean_squared_error(y_test, y_pred_linear))
    mae_linear_list.append(mean_absolute_error(y_test, y_pred_linear))
    mpae_linear_list.append(mean_percentage_absolute_error(y_test, y_pred_linear))

    # Modelo de Regressão Polinomial
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    modelo_polynomial = LinearRegression()
    modelo_polynomial.fit(X_train_poly, y_train)
    y_pred_poly = modelo_polynomial.predict(X_test_poly)
    mse_poly_list.append(mean_squared_error(y_test, y_pred_poly))
    mae_poly_list.append(mean_absolute_error(y_test, y_pred_poly))
    mpae_poly_list.append(mean_percentage_absolute_error(y_test, y_pred_poly))

# Calcular a média das métricas
mse_linear_media = np.mean(mse_linear_list)
mae_linear_media = np.mean(mae_linear_list)
mse_poly_media = np.mean(mse_poly_list)
mae_poly_media = np.mean(mae_poly_list)
mpae_linear_media = np.mean(mpae_linear_list) 
mpae_poly_media = np.mean(mpae_poly_list)

print(f'MSE Médio Modelo Linear: {mse_linear_media}')
print(f'MAE Médio Modelo Linear: {mae_linear_media}')
print(f'MSE Médio Modelo Polinomial: {mse_poly_media}')
print(f'MAE Médio Modelo Polinomial: {mae_poly_media}')
print(f'MPAE Médio Modelo Linear: {mpae_linear_media}')  
print(f'MPAE Médio Modelo Polinomial: {mpae_poly_media}')  

### SERIES TEMPORAIS - ARIMA - PREVISAO DE FROTA DE ELETRIFICADOS PARA OS PROXIMOS ANOS
### PREPARACAO DOS DADOS - PREVISAO DE FROTA DE ELETRIFICADOS PARA OS PROXIMOS ANOS

# Concatenando os DataFrames de cada ano em um único DataFrame
df_total = pd.concat([df18, df19, df20, df21, df22, df23])

# Filtrar para veículos elétricos e híbridos e agrupar por ano
df_filtrado = df_total[df_total['Combustivel_Veiculo'].isin(['HIBRIDO', 'HIBRIDO_PLUG_IN', 'ELETRICO','CELULA_COMBUSTIVEL'])]
df_anual = df_filtrado.groupby('Ano')['Qtd._Veiculos'].sum()


from statsmodels.tsa.arima.model import ARIMA # Importando ARIMA 
from statsmodels.tsa.stattools import adfuller # Teste para verificar se os dados sao estacionarios 

# Ajustar o modelo ARIMA, parametros p, d, q escolhidos com valores baixos(1, 1, 1) para uma primeira abordagem
model = ARIMA(df_anual, order=(1, 1, 1))
model_fit = model.fit()

# Fazer previsões para os próximos 3 anos, 3 steps
previsoes = model_fit.forecast(steps=3)

print("Previsões ARIMA:")
print(previsoes)

# Plotar as previsões
plt.figure(figsize=(10, 6))
plt.plot(df_anual.index, df_anual, label='Dados Históricos')
plt.plot([2024, 2025, 2026], previsoes, label='Previsões ARIMA', color='red')
plt.title('Previsão de Veículos Eletrificados para 2024, 2025 e 2026')
plt.xlabel('Ano')
plt.ylabel('Quantidade de Veículos')
plt.xticks(list(df_anual.index) + [2024, 2025, 2026])  # Adicionando anos nas previsões
# Adicionar rótulos com os valores numéricos
for year, value in zip(df_anual.index, df_anual):
    plt.text(year, value, f'{int(value)}', ha='center', va='bottom')
for year, value in zip([2024, 2025, 2026], previsoes):
    plt.text(year, value, f'{int(value)}', ha='center', va='bottom')
plt.legend()
plt.show()

# Avaliar o Modelo, df_anual = série temporal de dados históricos

# Definindo um tamanho de "janela" para walk-forward validation
window = 3  # 3 anos para treinar e prever o próximo ano

# Métrica MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    # Evitar divisão por zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    
    # Calcular MAPE somente para valores não-zero
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100


# Listas para armazenar os resultados das métricas
mse_values = [] # Mean Squared Error
mae_values = [] # Mean Absolute Error
mape_values = [] # Mean Absolute Percentage Error 

for i in range(window, len(df_anual)):
    train = df_anual.iloc[:i]
    test = df_anual.iloc[i:i+1]
    
    # Ajustar o modelo no conjunto de treino
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Prever o próximo ponto
    predictions = model_fit.forecast(steps=1)
    
    # Calcular e armazenar as métricas
    mse = mean_squared_error(test, predictions)
    mae = mean_absolute_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    mse_values.append(mse)
    mae_values.append(mae)
    mape_values.append(mape)

# Exibir a média das métricas
print(f"Média do MSE ARIMA: {sum(mse_values) / len(mse_values)}") # 121706901.5165714
print(f"Média do MAE ARIMA: {sum(mae_values) / len(mae_values)}") # 11032.085093787638
print(f"Média do MAPE ARIMA: {sum(mape_values) / len(mape_values)}") # 6.015904011183016
# 'Too few observations to estimate starting parameters

# Função para realizar e imprimir os resultados do teste ADF
def testar_estacionariedade(serie):
    resultado_adf = adfuller(serie, autolag='AIC')
    print(f'Estatística ADF : {resultado_adf[0]}') # 31.668190766272982
    print(f'p-valor : {resultado_adf[1]}') # 1.0
    for key, value in resultado_adf[4].items():
        print(f'Valor Crítico {key} : {value}') 
        # Valor Crítico 1% : -10.41719074074074
        # Valor Crítico 5% : -5.77838074074074
        # Valor Crítico 10% : -3.391681111111111

# Aplicar o teste ADF na série temporal
testar_estacionariedade(df_anual)

# Estatística ADF = 31.67, é muito maior que qualquer um dos valores críticos
# p-valor =1, logo a série temporal provavelmente não é estacionária, mesmo com 1 diferenciacao


### SARIMA  - PREVISAO DE FROTA DE ELETRIFICADOS PARA OS PROXIMOS ANOS

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Concatenando os DataFrames de cada ano em um único DataFrame
df_total = pd.concat([df18, df19, df20, df21, df22, df23])

# Filtrar para veículos elétricos e híbridos e agrupar por ano
df_filtrado = df_total[df_total['Combustivel_Veiculo'].isin(['HIBRIDO', 'HIBRIDO_PLUG_IN', 'ELETRICO','CELULA_COMBUSTIVEL'])]
df_anual = df_filtrado.groupby('Ano')['Qtd._Veiculos'].sum()

# Dividir os dados em conjunto de treino e teste 
train = df_anual.iloc[:-3]
test = df_anual.iloc[-3:]

# Definir os parâmetros do modelo SARIMA
# Os valores p, d, q são para a parte ARIMA do modelo
# Os valores P, D, Q, s são para a parte sazonal
# P = olha para o valor da série temporal em uma temporada anterior para ajudar a prever o valor atual.
# D = modelo aplica diferenciação sazonal
# Q = modelo leva em conta quanto ele errou na última temporada ao fazer previsões para esta temporada.
# SARIMAX(df_anual, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)), s = 12, pois a sazonalidade é anual (12 meses)
sarima_model = SARIMAX(df_anual, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
# como observado no ARIMA, devido ao volume de dados, a diferenciacao sazonal (D=0) nao foi aplicada
sarima_model_fit = sarima_model.fit()

# Previsoes
sarima_previsoes = sarima_model_fit.forecast(steps=len(test))

print("Previsões SARIMA:")
print(sarima_previsoes)

# Plotar as previsões junto com os dados históricos
plt.figure(figsize=(10, 6))
plt.plot(df_anual.index, df_anual, label='Dados Históricos')
plt.plot([2024, 2025, 2026], sarima_previsoes, label='Previsões SARIMA', color='orange')
plt.title('Previsão de Veículos Eletrificados para 2024, 2025 e 2026')
plt.xlabel('Ano')
plt.ylabel('Quantidade de Veículos')
plt.xticks(list(df_anual.index) + [2024, 2025, 2026])  # Adicionando anos nas previsões
# Adicionar rótulos com os valores numéricos
for year, value in zip(df_anual.index, df_anual):
    plt.text(year, value, f'{int(value)}', ha='center', va='bottom')
for year, value in zip([2024, 2025, 2026], sarima_previsoes):
    plt.text(year, value, f'{int(value)}', ha='center', va='bottom')
plt.legend()
plt.show()

# Definicao da funcao para calculo do MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100

# Etapa de validação cruzada
mse_scores = []
mae_scores = []
mape_scores = []

for t in range(1, len(df_anual) - 2):
    # Divisão em treino e teste
    train = df_anual.iloc[:t]
    test = df_anual.iloc[t:t+1]
    
    # Treinando o modelo
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
    model_fit = model.fit(disp=False)
    
    # Previsão
    pred = model_fit.forecast(steps=len(test))
    
    # Calculando o MSE, MAE e MAPE para validação cruzada
    mse = mean_squared_error(test, pred)
    mae = mean_absolute_error(test, pred)
    mape = mean_absolute_percentage_error(test, pred)
    mse_scores.append(mse)
    mae_scores.append(mae)
    mape_scores.append(mape)

# Calculando a média dos erros
mean_mse = sum(mse_scores) / len(mse_scores)
mean_mae = sum(mae_scores) / len(mae_scores)
mean_mape = sum(mape_scores) / len(mape_scores)

print(f'Média do MSE SARIMA: {mean_mse}') # Média do MSE: 853866841.0
print(f'Média do MAE SARIMA: {mean_mae}') # Média do MAE: 29221.0 
print(f'Média do MAPE SARIMA: {mean_mape}') # Média do MAPE: 44.88770776367937

### Facebook Prophet  - PREVISAO DE FROTA DE ELETRIFICADOS PARA OS PROXIMOS ANOS

from prophet import Prophet # biblioteca Facebook Prophet
import matplotlib.dates as mdates # Biblioteca para alteração do formato das datas no plot

# Concatenando os DataFrames de cada ano em um único DataFrame
df_total = pd.concat([df18, df19, df20, df21, df22, df23])

# Filtrar para veículos elétricos e híbridos e agrupar por ano
df_filtrado = df_total[df_total['Combustivel_Veiculo'].isin(['HIBRIDO', 'HIBRIDO_PLUG_IN', 'ELETRICO','CELULA_COMBUSTIVEL'])]
df_anual = df_filtrado.groupby('Ano')['Qtd._Veiculos'].sum()

# Primeiro, preparando os dados para o formato que o Prophet espera
df_prophet = df_anual.reset_index()
df_prophet.columns = ['ds', 'y']
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y') + pd.offsets.YearBegin(0)  # padronizando as datas para o início do ano para evitar previsões no mesmo ano 

# Instanciando e ajustando o modelo do Prophet
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(df_prophet)

# Criando um DataFrame futuro para as previsões
futuro = model.make_future_dataframe(periods=4, freq='Y')
futuro['ds'] = futuro['ds'].dt.to_period('Y').dt.to_timestamp()

# Fazendo as previsões
previsao = model.predict(futuro)
print(previsao)

# Filtrando as previsões para os anos de interesse (2024, 2025, 2026) e exibir
previsao_filtrado = previsao[(previsao['ds'].dt.year >= 2024) & (previsao['ds'].dt.year <= 2026)][['ds', 'yhat']]
print(previsao_filtrado)

# Filtrando somente as previsões para usar no gráfico
previsao_plot = previsao[previsao['ds'].dt.year >= 2024]

# o Prophet gera o nome das legendas automaticamente, por padrão, em inglês
# Plotando as previsões manualmente e alterando os nomes das legendas
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(previsao_plot['ds'], previsao_plot['yhat'], color='red', label='Previsão', marker='x')
ax.fill_between(previsao_plot['ds'], previsao_plot['yhat_lower'], previsao_plot['yhat_upper'], color='gray', alpha=0.5, label='Limite de Confiança')
ax.plot(df_prophet['ds'], df_prophet['y'], label='Dados Reais', color='blue', marker='o')

# Adicionando título e rótulos personalizados ao gráfico
plt.title('Previsões de Veículos Híbridos e Elétricos para 2024, 2025 e 2026')
plt.xlabel('Ano')
plt.ylabel('Quantidade de Veículos')
plt.legend() # Mostrar legenda
plt.show() # Exibir o gráfico

# Avaliando o modelo, etapa de validação cruzada:
from prophet.diagnostics import cross_validation, performance_metrics

# Ajustando o modelo
model = Prophet(yearly_seasonality=True)
model.fit(df_prophet)

# Definindo os parâmetros para a validação cruzada
# horizon: até quanto tempo no futuro queremos prever
# initial: tamanho da janela inicial de treinamento
# period: frequência com que são feitos os cortes transversais
df_cv = cross_validation(model, horizon='365 days', period='180 days', initial='730 days')

# Calculando métricas de desempenho
df_p = performance_metrics(df_cv)

# Exibindo as métricas
print(df_p) #mae = 24905.2, MAPE = 13,58%

### RESUTLADOS DAS PREVISÕES ###

# Médias das acurácias anotadas de cada modelo
MAE = {
    'Modelo': ['Regressão Linear', 'Polinomial', 'ARIMA', 'SARIMA', 'Prophet'],
    'MAE': [27745, 18030, 11032, 29221, 24905]
}

# Convertendo em um DataFrame
df_mae = pd.DataFrame(MAE)

# Criando o gráfico de barras com Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Modelo', y='MAE', data=df_mae, palette='pastel')

# Adicionando títulos e rótulos
plt.title('Erro absoluto médio (MAE) dos Modelos de Séries Temporais')
plt.xlabel('Modelos')
plt.ylabel('MAE')
plt.xticks(rotation=45)  # Rotação dos nomes dos modelos para melhor leitura

# Exibindo o gráfico
plt.show()
