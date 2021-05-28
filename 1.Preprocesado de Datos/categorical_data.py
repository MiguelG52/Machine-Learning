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

#tratamiento de NAS
from sklearn.impute import SimpleImputer
#axis = o pide datos de columna, axis = 1 de fila
imp = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
imp = SimpleImputer().fit(x[:, 1:3])
x[:, 1:3] = imp.transform(x[:, 1:3])


#datos categoricos pero ya es obsoleto
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
le_x = preprocessing.LabelEncoder()
x[:, 0] = le_x.fit_transform(x[:,0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)

x = np.array(ct.fit_transform(x), dtype = np.float)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#escalado de variables por normalizacion
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train =  sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
