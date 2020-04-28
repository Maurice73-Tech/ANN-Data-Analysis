#Import von notwendigen libraries und Tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data needs to be imported
dataset = pd.read_csv("component_faults_data.csv")
dataset.head(29256) 

#Pandas Datenstruktur (siehe oben) wird zu numpy arrays geändert 
#Changing pandas dataframe to numpy array
X = dataset.iloc[:,:48].values #attribut werte 
y = dataset.iloc[:,48:49].values # Class Labels

# Normalisierung der Daten mit StandardScaler
# hierbei wird jedem Wert, der Meanvalue der jeweiligen Spalte abgezogen. Anschliessend wird der Wert mit der Varianz skaliert
# src: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

print(X)

#one hot encoding um Labels in Array dar zu stellen, labels in 0 und 1 darstellen
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# split data 90% training and 10% test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1, random_state=1)
# zweiter Data split 90% des übrig gebliebenen training sets in 90% training 10% validation set 
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.1, random_state=1)


# Notwendige imports zur ANN Bildung
import keras
from keras.models import Sequential
from keras.layers import Dense

# ANN layer werden initialisiert 
# 1. Layer (Input Layer) 48 Neuronen da 48 Attribute 2. Layer (hidden layer) 22 Neuronen da konstant gute Ergebnisse
# 3. Layer (Output Layer) 11 Neuronen da 11 Klassen 
model = Sequential()
model.add(Dense(22, input_dim=48, activation='relu')) 
model.add(Dense(11, activation= 'softmax'))


# loss function und optimizer
# Kategorische Kreuzentropy verwendet einfacherer Kennzahlen für Genauigkeit,
# wie die mittlere quadratische Abweichung oder der mittlere absolute prozentuale Fehler
model.compile(loss="categorical_crossentropy", optimizer="adam" , metrics = ["accuracy"])


#training Model result  example batch size to partially compute data 
# validations et to get an insight regarding overfitting

results = model.fit(X_train, y_train, validation_data = (X_val,y_val), epochs=100, batch_size=64)

# Nun wird mithilfe der Predict Methode getestet 
# Evaluate wurde ebenfalls verwendet, da wir allerdings keinen Unterschied feststellen konnten 
# verwendeten wir weiterhin predict
y_pred = model.predict (X_test)

# Nach dem test der Attribute wird eine Liste der Labels erstellt
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
# Anschliessend werde die oneHot encoded Labels wieder in ihre uhrsprüngliche Form konvertiert     
# = Reverse oneHot encoding
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
# zuletzt wird die Accuracy in 100% berechnet und ausgegeben   
from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

# in diesem Plot wird der Verlauf der Trainings-Accuracy mit der Accuracy auf dem Validation data set verglichen
# Dadurch lässt sich over- und underfitting frühzeitig erkennen 
import matplotlib.pyplot as plt
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# in diesem Plot wird der Verlauf der Trainings-Loss mit dem loss auf dem Validation data set verglichen
# hat denselben Zweck wie oben
plt.plot(results.history['loss']) 
plt.plot(results.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

