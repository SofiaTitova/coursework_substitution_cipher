import numpy as np
from pycipher import SimpleSubstitution as SimpSub
import math
import random
import re
import copy
import matplotlib.pyplot as plt

# Расчет расстояния Хемминга
def hamming_distance(str_1,str_2):
    distance = 0
    if (len(str_1) == len(str_2)):
        for i in range (len(str_1)):
            if str_1[i] != str_2[i]:
                distance += 1


    return distance

def inverse_substition_key(key):
    inverse_key = copy.copy(key)
    for i in range (len(key)):
        inverse_key[ord(key[i]) - 65] = chr(i + 65)
    return inverse_key

def frequency_ngram_score(dict_frequency_test, dict_frequency_training):
    score = 0
    l = len(dict_frequency_training)
    for elem in dict_frequency_training:
        score += np.abs(dict_frequency_test[elem] - dict_frequency_training[elem])
    return score

def identityscore(true_inverse_key, our_inverse_key):
    score = 0
    for i in range (len(true_inverse_key)):
        if (true_inverse_key[i] == our_inverse_key[i]):
            score +=1
    return score

def swap(S, i, j):
    L = list(S)
    L[i], L[j] = L[j], L[i]
    S = "".join(L)
    return S


def getFrequencyDict(text):
    text = text.replace(" ","")
    D = {chr(i+65):0 for i in range(26)}
    for c in text:
        D[c] += 1
    for i in range(26):
        D[chr(i+65)] =D[chr(i+65)]/len(text)
    return D

def dict_to_file(D, file_name):
    file = open(file_name,"w",encoding="utf8")
    for key, values in D.items():
        file.write(str(key) + " " +str (values)+"\n")
    file.close

# Логарифм вероятности для примитивной марковской цепи на n-граммах
#
def marcov_chain_score(deciphering_test_text, dict_frequency_training, n):
    score = 0
    for i in range (0, len(deciphering_test_text) - n + 1):
        c = deciphering_test_text[i:i+n]
        if (c in dict_frequency_training):
            score += np.log10(dict_frequency_training[c])
        else:
            score += 10^(-n)
    return score
# Чтение и очистка
file_training = open("training_text.txt","r",encoding="utf8")
text_training = file_training.read()
text_training = "".join(text_training.splitlines())
text_training = re.sub('[^A-Z] ','',text_training.upper())

file_test = open("test_text.txt","r",encoding="utf8")
text_test = file_test.read()
text_test= "".join(text_test.splitlines())
text_test = re.sub('[^A-Z] ','',text_test.upper())
print(text_test[0:100])
#########################
text_training_part = text_training[0:100]
text_test_part = text_test[0:100]
random.seed(1)
key = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
random.shuffle(key)
print(key)
enciphered_text_test_part = SimpSub(key).encipher(text_test_part,True)

deciphered_text_test = SimpSub(key).decipher(enciphered_text_test_part,True)

print(deciphered_text_test)
############################
text_test_part = text_test[0:100]
enciphered_text_test = SimpSub(key).encipher(text_test_part,True)
truekey = copy.copy(key)

dict_frequency_training = getFrequencyDict(text_training_part)
dict_to_file(dict_frequency_training,"test_dict_freq.txt")

N = 1000
identity_score_array = np.zeros(N)
frequency_score_array = np.zeros(N)
hamming_distance_array = np.zeros(N)
marcov_chain_score_array = np.zeros(N)

max_score = 0
max_key = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
random.shuffle(max_key)
for i in range (N):
     ### hill climbing
    #здесь меняем клююч попарно
    a = random.randint(0, 25)
    b = random.randint(0, 25)
    key = swap(max_key, a, b)
    deciphered_text_test = SimpSub(key).decipher(enciphered_text_test,True)
    dict_frequency_test = getFrequencyDict(deciphered_text_test)
    score = identityscore(truekey, key)
    if (score > max_score):
        max_score = score
        max_key = copy.copy(key)
    # print(truekey,key)
    # print(identityscore(truekey, key))
    identity_score_array[i]= identityscore(truekey, key)
    frequency_score_array[i] = frequency_ngram_score(dict_frequency_test,dict_frequency_training)
    hamming_distance_array[i] = hamming_distance(deciphered_text_test, text_test_part)
    marcov_chain_score_array[i] = marcov_chain_score(deciphered_text_test, dict_frequency_training, 1)

################## plotting

fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(10, 5))
ax1.plot(identity_score_array, linewidth=2.0)
ax2.plot(frequency_score_array, linewidth=2.0)
ax3.plot(hamming_distance_array, linewidth=2.0)
ax4.plot(marcov_chain_score_array, linewidth=2.0)
plt.show()
file_training.close()
file_test.close()


