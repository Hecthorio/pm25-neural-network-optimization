# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 18:07:08 2025

@author: hecto
"""

'''
Script para evaluar si existe diferencia significativa entre los modelo de RN
a partir de un ANOVA
'''

#librerias
import pandas as pd
from scipy import stats

#definimos la ruta donde estan los mse de los modelos
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/3er Congreso internacional multidisciplinario de la divulgación científica/'

#leemos el archivo y generamos el df
df = pd.read_csv(ruta + 'resultados_mse_training.csv')

# Separamos los MSE de cada modelo en listas
mse_fnn = df['FNN']
mse_rnn = df['RNN']
mse_lstm = df['LSTM']
mse_gru = df['GRU']
mse_cnn = df['CNN']

# Aplicamos ANOVA de una vía
f_statistic, p_value = stats.f_oneway(mse_fnn, mse_rnn, mse_lstm, mse_gru, mse_cnn)

# Mostramos los resultados
print(f"F = {f_statistic:.4f}, p = {p_value:.4f}")

# Interpretación básica
alpha = 0.05
if p_value < alpha:
    print("Hay diferencias significativas entre los modelos.")
else:
    print("No hay diferencias significativas entre los modelos.")
    
# import seaborn as sns
# import matplotlib.pyplot as plt

df_mod = df.melt(var_name='Modelo', value_name='MSE')

# sns.boxplot(x='Modelo', y='MSE', data=df_mod)
# plt.title("Distribución de MSE por Modelo")
# plt.show()

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Aplicar prueba de Tukey HSD
tukey = pairwise_tukeyhsd(endog=df_mod['MSE'], groups=df_mod['Modelo'], alpha=0.05)

# Mostrar resumen de resultados
print(tukey.summary())

#convertimos los resultados a un df para poder pasarlo a la función letters
tukey_result = pd.DataFrame(data=tukey.summary().data[1:],  # Saltamos la fila de encabezados [1:]
                            columns=tukey.summary().data[0])  # Usamos los e

#para generar las retras en los encabezados a partir de la siguiente función
#con ayuda de la libreria string
import string

#hay que agradecer a nuestro compa el indu por la función xdxdxd, se la rifo
def letters(df, alpha=0.05):

    df["p-adj"] = df["p-adj"].astype(float)

    # Creating a list of the different treatment groups from Tukey's
    group1 = set(df.group1.tolist())  # Dropping duplicates by creating a set
    group2 = set(df.group2.tolist())  # Dropping duplicates by creating a set
    groupSet = group1 | group2  # Set operation that creates a union of 2 sets
    groups = list(groupSet) #removed sorted from here

    # Creating lists of letters that will be assigned to treatment groups
    letters = list(string.ascii_lowercase)[:len(groups)]
    cldgroups = letters

    # the following algoritm is a simplification of the classical cld,

    cld = pd.DataFrame(list(zip(groups, letters, cldgroups)))
    cld[3]=""
    
    for row in df.itertuples():
        if df["p-adj"][row[0]] > (alpha):
            cld.iat[groups.index(df["group1"][row[0]]), 2] += cld.iat[groups.index(df["group2"][row[0]]), 1]
            cld.iat[groups.index(df["group2"][row[0]]), 2] += cld.iat[groups.index(df["group1"][row[0]]), 1]
            
        if df["p-adj"][row[0]] < (alpha):
                cld.iat[groups.index(df["group1"][row[0]]), 3] +=  cld.iat[groups.index(df["group2"][row[0]]), 1]
                cld.iat[groups.index(df["group2"][row[0]]), 3] +=  cld.iat[groups.index(df["group1"][row[0]]), 1]

    cld[2] = cld[2].apply(lambda x: "".join(sorted(x)))
    cld[3] = cld[3].apply(lambda x: "".join(sorted(x)))
    cld.rename(columns={0: "groups"}, inplace=True)

    # this part will reassign the final name to the group
    # for sure there are more elegant ways of doing this
    cld = cld.sort_values(cld.columns[2], key=lambda x: x.str.len())
    cld["labels"] = ""
    letters = list(string.ascii_lowercase)
    unique = []
    for item in cld[2]:

        for fitem in cld["labels"].unique():
            for c in range(0, len(fitem)):
                if not set(unique).issuperset(set(fitem[c])):
                    unique.append(fitem[c])
        g = len(unique)

        for kitem in cld[1]:
            if kitem in item:
                if cld["labels"].loc[cld[1] == kitem].iloc[0] == "":
                    cld["labels"].loc[cld[1] == kitem] += letters[g]

                #Checking if there are forbidden pairing (proposition of solution to the imperfect script)                
                if kitem in ' '.join(cld[3][cld["labels"]==letters[g]]): 
                    g=len(unique)+1
               
                # Checking if columns 1 & 2 of cld share at least 1 letter
                if len(set(cld["labels"].loc[cld[1] == kitem].iloc[0]).intersection(cld.loc[cld[2] == item, "labels"].iloc[0])) <= 0:
                    if letters[g] not in list(cld["labels"].loc[cld[1] == kitem].iloc[0]):
                        cld["labels"].loc[cld[1] == kitem] += letters[g]
                    if letters[g] not in list(cld["labels"].loc[cld[2] == item].iloc[0]):
                        cld["labels"].loc[cld[2] == item] += letters[g]

    cld = cld.sort_values("labels")

    cld.drop(columns=[1, 2, 3], inplace=True)
    cld= dict(zip(cld["groups"], cld["labels"]))

    return(cld)

import matplotlib.pyplot as plt
import numpy as np

#evaluamos algunas cosas para la grafica de barras
df_bar = df_mod.groupby('Modelo')['MSE'].agg(['mean','sem']).reset_index()

#acomodamos los valores del df del mayor al menor
df_bar = df_bar.sort_values(by='mean', ascending = False)

#llamamos la función para generar las etiquetas
group_labels = letters(tukey_result)

plt.figure(figsize=(12, 10), dpi=300)
error = np.full(len(df_bar), df_bar['sem'])
custom_letters = group_labels

# Create the bar plot
bars = plt.bar(df_bar['Modelo'], df_bar['mean'], yerr=error, capsize=5)

# Add annotations above bars
for bar, modelo in zip(bars, df_bar['Modelo']):
    height = bar.get_height()
    plt.annotate(
        custom_letters[modelo],
        xy=(bar.get_x() + bar.get_width() / 2, height + [0.001 if modelo == 'CNN' else 0][0]), #OJO checar el valor que se suma porque puede salirse del área de la grafica las letras
        xytext=(0, 5),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom'
    )

# Set x-ticks with rotation
plt.xticks(
    ticks=range(len(df_bar['Modelo'])),
    labels=df_bar['Modelo'],
    rotation=45,
    ha='right'
)

# Add labels and title with larger font sizes and spacing
plt.xlabel('Modelos', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.ylim(0,0.011)
plt.title('Modelos de RN para pronostico de PM$_{2.5}$', fontsize=16, pad=20)  # Increased title font size and added padding

# Adjust layout for better spacing
plt.tight_layout()
plt.show()