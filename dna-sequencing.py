
from Bio import SeqIO

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# un exemple de séquence ADN
for sequence in SeqIO.parse('data/dna-sequence-dataset/example_dna.fa', "fasta"):
    print("ID de la séquence :",sequence.id)
    print("Taille de la séquence :",len(sequence))
    print("-> [ {} ]".format(sequence.seq))
    break

def chaine_vers_tableau(sequence):
   sequence = sequence.lower()
   sequence = re.sub('[^acgt]', '', sequence)
   sequence = np.array(list(sequence))
   return sequence

# création d'un encodeur à label avec l'alphabet {a,c,g,t}
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t']))

def enc_ordinal(tableau):
    entier_encode = label_encoder.transform(tableau)
    flottant_encode = entier_encode.astype(float)
    flottant_encode[flottant_encode == 0] = 0.25 A
    flottant_encode[flottant_encode == 1] = 0.50 C
    flottant_encode[flottant_encode == 2] = 0.75 G
    flottant_encode[flottant_encode == 3] = 1.00 T
    flottant_encode[flottant_encode == 4] = 0.00 n'importe quelle autre lettre
    return flottant_encode

# testons le programme sur une courte séquence :
sequence_test = 'TTCAGCCAGTG'
enc_ordinal(chaine_vers_tableau(sequence_test))

def enc_one_hot(chaine):
    entier_encode = label_encoder.transform(chaine)
    onehot_encodeur = OneHotEncoder(sparse=False, dtype=int)
    entier_encode = entier_encode.reshape(len(entier_encode), 1)
    onehot_encode = onehot_encodeur.fit_transform(entier_encode)
    onehot_encode = np.delete(onehot_encode, -1, 1)
    return onehot_encode

# testons le programme sur une courte séquence :
sequence_test = 'GAATTCTCGAA'
enc_one_hot(chaine_vers_tableau(sequence_test))

def kmers(sequence, k):
    return [sequence[indice:indice+k].upper() for indice in range(len(sequence) - k + 1)]

# testons le programme sur une courte séquence :
sequence_test = 'ATGCATGCA'
kmers(sequence_test,5)

mots = kmers(sequence_test, 5)
phrase = ' '.join(mots)
phrase

sequence1 = 'TCTCACACATGTGCCAATCACTGTCACCC'
sequence2 = 'GTGCCCAGGTTCAGTGAGTGACACAGGCAG'
phrase1 = ' '.join(kmers(sequence1, 6))
phrase2 = ' '.join(kmers(sequence2, 6))

# on construit ensuite le modèle de sac-de-mots
cv = CountVectorizer()
X = cv.fit_transform([phrase, phrase1, phrase2]).toarray()
X

adn_humain = pd.read_table('data/dna-sequence-dataset/human.txt')
adn_humain.head()

adn_humain['class'].value_counts().sort_index().plot.bar()
plt.title("Distribution des classes dans l'ADN humain")

adn_chimp = pd.read_table('data/dna-sequence-dataset/chimpanzee.txt')
adn_chimp.head()

adn_chimp['class'].value_counts().sort_index().plot.bar()
plt.title("Distribution des classes dans l'ADN des chimpanzés")

adn_chien = pd.read_table('data/dna-sequence-dataset/dog.txt')
adn_chien.head()

adn_chien['class'].value_counts().sort_index().plot.bar()
plt.title("Distribution des classes dans l'ADN des chiens")

def kmers(seq, size=6):
    return [seq[x:x+size].upper() for x in range(len(seq) - size + 1)]

# nos séquences utilisées comme données d'entraînement sont converties en hexamères
# on applique ensuite le même procédé aux espèces différentes

adn_humain['words'] = adn_humain.apply(lambda x: kmers(x['sequence']), axis=1)
adn_humain = adn_humain.drop('sequence', axis=1)

adn_chimp['words'] = adn_chimp.apply(lambda x: kmers(x['sequence']), axis=1)
adn_chimp = adn_chimp.drop('sequence', axis=1)

adn_chien['words'] = adn_chien.apply(lambda x: kmers(x['sequence']), axis=1)
adn_chien = adn_chien.drop('sequence', axis=1)

adn_humain.head()

humain_texte = list(adn_humain['words'])
for i in range(len(humain_texte)):
    humain_texte[i] = ' '.join(humain_texte[i])

chimp_texte = list(adn_chimp['words'])
for i in range(len(chimp_texte)):
    chimp_texte[i] = ' '.join(chimp_texte[i])

chien_texte = list(adn_chien['words'])
for i in range(len(chien_texte)):
    chien_texte[i] = ' '.join(chien_texte[i])

# les labels sont séparés
y_humain = adn_humain.iloc[:, 0].values
y_chimp = adn_chimp.iloc[:, 0].values
y_chien = adn_chien.iloc[:, 0].values

y_humain

cv = CountVectorizer(ngram_range=(4,4))
X_humain = cv.fit_transform(humain_texte)
X_chimp = cv.transform(chimp_texte)
X_chien = cv.transform(chien_texte)

print("Pour les humains, on a {} gènes convertis en vectors de taille uniforme pour le comptage des hexamères 4-grammes.\n".format(X_humain.shape[0]))
print("Pour les chimpanzés, on a {} gènes convertis en vectors de taille uniforme pour le comptage des hexamères 4-grammes\n".format(X_chimp.shape[0]))
print("Pour les chiens, on a {} gènes convertis en vectors de taille uniforme pour le comptage des hexamères 4-grammes.".format(X_chien.shape[0]))

# on sépare les données d'ADN humain en sous-ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_humain, y_humain, test_size = 0.20, random_state=42)

classifieur = MultinomialNB(alpha=0.1)
classifieur.fit(X_train, y_train)

y_pred = classifieur.predict(X_test)

print("Matrice de confusion pour des prédictions sur des séquences d'ADN humain :\n")
print(pd.crosstab(pd.Series(y_test, name='Réalité'), pd.Series(y_pred, name='Prédiction')))

def métriques(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = métriques(y_test, y_pred)

print("\nexactitude = {} \nprécision  = {} \nrappel     = {} \nscore F1   = {}".format(accuracy, precision, recall, f1))

y_pred_chimp = classifieur.predict(X_chimp)

print("Matrice de confusion pour des prédictions sur des séquences d'ADN de chimpanzés :\n")
print(pd.crosstab(pd.Series(y_chimp, name='Réalité'), pd.Series(y_pred_chimp, name='Prédiction')))

accuracy, precision, recall, f1 = métriques(y_chimp, y_pred_chimp)

print("\nexactitude = {} \nprécision  = {} \nrappel     = {} \nscore F1   = {}".format(accuracy, precision, recall, f1))

y_pred_chien = classifieur.predict(X_chien)

print("Matrice de confusion pour des prédictions sur des séquences d'ADN de chiens :\n")
print(pd.crosstab(pd.Series(y_chien, name='Actual'), pd.Series(y_pred_chien, name='Predicted')))

accuracy, precision, recall, f1 = métriques(y_chien, y_pred_chien)

print("\nexactitude = {} \nprécision  = {} \nrappel     = {} \nscore F1   = {}".format(accuracy, precision, recall, f1))
