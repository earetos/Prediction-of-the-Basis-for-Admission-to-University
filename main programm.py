# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:31:12 2023

@author: User
"""


############ Libraries ###############
import pandas as pd
import tensorflow as tf
import keras
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import statistics
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
import math
a="train math Athens.csv"
b="train eng Thess.csv"
c="train eng Patra.csv"
d="train_fisiki_Ioannina.csv"

print("ΜΠΟΡΕΙΤΕ ΝΑ ΕΠΙΛΕΞΕΤΕ ΓΙΑ ΠΟΙΟ ΤΜΗΜΑ ΑΠΟ ΤΑ ΠΑΡΑΚΑΤΩ ΕΠΙΘΥΜΗΤΕ ΝΑ ΒΡΕΙΤΕ ΤΗΝ ΒΑΣΗ ΕΙΣΑΓΩΓΗΣ")
print("Πληκτρολογήστε τον αντίστοιχο αριθμό του κάθε τμήματος:")
print("Πληκτρολογήστε 1 για το Μαθηματικό Αθηνών \n 2 για τους Ηλεκτρολόγους μηχανικούς και μηχανικούς υπολογιστών της Θεσσαλονίκης",
      "\n 3 για τους Μηχανολόγους και Αεροναυπηγούς μηχανικούς Πατρων και τέλος \n 4 για το Φυσικό Ιωαννίνων:")
test=0
tmima="0"
while test==0:
    epilogi=str(input("Επιλέξτε τμήμα:"))

    if epilogi=="1":
        tmima=a
        test=1
    else:
        if epilogi=="2":
            tmima=b
            test=1
        else:
            if epilogi=="3":
                tmima=c
                test=1
            else:
                    if epilogi=="4":
                        tmima=d
                        test=1
                    else: 
                            print("Λανθασμένη επιλογή.Δοκιμάστε ξανα")
                            test=0

pred_tmimatos=np.array([[2,38.15,43.19,10.64,5.25,0.76,17.78,26.44,18.41,9.78,16.84,10.74,16.32,16.55,14.14,9.58,25.86,17.54,13.75,17,15.32,10.7,27.03,16.22]])




#εισαγωγή του αρχείου των δεδομένων
dataset = pd.read_csv(tmima,sep=";",header=None)
stats_of_dataset=dataset.describe(include='all')
#print (dataset.describe(include='all'))

#Διαχωρισμός δεδομένων εισόδου και δεδομένων εξόδου
X=dataset.iloc[:,0:24].values
y=dataset.iloc[:,24].values

#Εκτύπωση των δυο συνόλων
print("Το σύνολο με τα στατιστικα των βαθμολογιών\n",X,"\n\nΤο σύνολο με τις βάσεις εισαγωγής\n",y)
print("\n@@@@@@@@@@@@@@@@@@@@@@@\n\nΠΑΡΑΚΑΛΩ ΠΕΡΙΜΕΝΕΤΕ ΜΕΧΡΙ ΝΑ ΟΛΟΚΛΗΡΩΘΕΙ Η ΔΙΑΔΙΚΑΣΙΑ!!!\n\n@@@@@@@@@@@@@@@@@@@@@@@\n")
           
# Create model
model_aprox = Sequential()
model_aprox.add(Dense(10, activation='relu', input_dim=24))
model_aprox.add(Dense(10, activation='relu'))
model_aprox.add(Dense(1))
            
model_aprox.compile(optimizer="adam", loss="mae",metrics=['mae'])
            
monitor_aprox = EarlyStopping(monitor='val_mae', patience=50, 
                    verbose=0, mode='min', restore_best_weights=True)
# Train model
model_aprox.fit(X, y, validation_split = 0.2, callbacks=[monitor_aprox],verbose=0,epochs=2000, batch_size=3)
       
        
#prediction
y_pred = model_aprox.predict(pred_tmimatos)

print("\n\n$$$$$$$$$$$$$$$$$$$\n\nΠΡΟΒΛΕΨΗ ΒΑΣΗΣ:",y_pred[0][0],"\n\n$$$$$$$$$$$$$$$$$$$$$$$$\n")           
            
