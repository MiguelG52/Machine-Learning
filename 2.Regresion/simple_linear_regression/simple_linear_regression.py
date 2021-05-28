#@author: miguelG52

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importa librerias
dataset = pd.read_csv('Salary_Data.csv')

#importar dataset
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

#dividdir data set en conjunto de entrenamiento
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =1/3, random_state=0 )

#Crear modelo de regresion lineal simple
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

#predecir el conjunto de test
y_pred = regression.predict(x_test)



#visualizar los resultados de test

#visualizar los resultados de entrenamiento
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')
plt.title("Sueldo vs Años de experiencia")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo ($)")
plt.show()