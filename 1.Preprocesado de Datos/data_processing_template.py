#plantilla de procesado
#@author: MiguelG52


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importamos Data set
#se deve de tener un archivo en el mismo directorio del scrip
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


#dividir dataset  en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#escalado de variables por normalizacion
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train =  sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
