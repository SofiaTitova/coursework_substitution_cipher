import numpy as np
from pycipher import SimpleSubstitution as SimpSub
import math
import random
import re
import copy
import matplotlib.pyplot as plt

def select_metric(max_score):
    print("Выберите метрику:")
    print("1. Частотный анализ n-грамм")
    print("2. Расстояние Хемминга")
    print("3. Марковская цепь")
    print("4. Идентичность ключей")

    choice = int(input("Введите номер выбранной метрики: "))

    match choice:
        case 1:
            max_score = np.inf  # frequency_ngram_score
        case 2:
            max_score = np.inf  # hamming_distance(
        case 3:
            max_score = - np.inf  # marcov_chain_score
        case 4:
            max_score = - np.inf  # identityscore

    return int(choice), max_score

def use_score(i, dict_frequency_test, dict_frequency_training, deciphered_text_test, key, truekey, text_test_part, n):
    match i:
        case 1:
            return frequency_ngram_score(dict_frequency_test, dict_frequency_training)
        case 2:
            return hamming_distance(deciphered_text_test, text_test_part)
        case 3:
            return marcov_chain_score(deciphered_text_test, dict_frequency_training, n)
        case 4:
            return identityscore(truekey, key)
# Расчет расстояния Хемминга
def hamming_distance(str_1,str_2):
    distance = 0
    if (len(str_1) == len(str_2)):
        for i in range (len(str_1)):
            if str_1[i] != str_2[i]:
                distance += 1
    return distance

def frequency_ngram_score(dict_frequency_test, dict_frequency_training):
    score = 0
    l = len(dict_frequency_training)
    for elem in dict_frequency_test:
        if dict_frequency_training.get(elem) is not None:
            score += np.abs(dict_frequency_test[elem] - dict_frequency_training[elem])
        else:
            score += dict_frequency_test[elem]
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

def get_n_gramm(n, text):
    D = {}
    for i in range (len(text) - n + 1):
        if D.get(text[i:i+n]) is None:
            D[text[i:i+n]] = 1
        else:
            D[text[i:i+n]] += 1
    for elem in D:
        D[elem] = D[elem]/(len(text) - n + 1)
    return D

def dict_to_file(D, file_name):
    file = open(file_name,"w",encoding="utf8")
    for key, values in D.items():
        file.write(str(key) + " " +str (values)+"\n")
    file.close

# Логарифм вероятности для примитивной марковской цепи на n-граммах
def marcov_chain_score(deciphering_test_text, dict_frequency_training, n):
    score = 0
    for i in range (0, len(deciphering_test_text) - n + 1):
        c = deciphering_test_text[i:i+n]

        if (c in dict_frequency_training):
            score += np.log10(dict_frequency_training[c])
        else:
            score += np.log10(min_p/100)
    return score

def text_clean_and_read(name):
    file_tmp = open(name, "r", encoding="utf8")
    text_tmp = file_tmp.read()
    text_tmp = " ".join(text_tmp.splitlines())
    text_tmp = re.sub('[^A-Z ]', '', text_tmp.upper())
    file_tmp.close()
    return text_tmp


# Функция шифрования по ключу
# text - list символов текста
# key - list с перестановкой на алфавите
def encrypt(text, key):
    result = []
    for i in range(len(text)):
        result.append(key[alphabet.index(text[i])])

    return ''.join(map(str, result))


# Функция дешифрования по ключу
# code - list символов шифра
# key - list с перестановкой на алфавите
def decrypt(code, key):
    result = []
    for i in range(len(code)):
        result.append(alphabet[key.index(code[i])])
    return ''.join(map(str, result))

text_training = text_clean_and_read("new_training_text.txt")
text_test = text_clean_and_read("text_to_decode.txt")
text_training_part = text_training #[0:10000]
text_test_part = text_test #[0:text_len_part]
seeD = int(input("enter the seed value: "))
random.seed(seeD)
alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
key = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
random.shuffle(key)
enciphered_text_test_part = encrypt(text_test_part,key)
print(enciphered_text_test_part)
deciphered_text_test = decrypt(enciphered_text_test_part,key)
print(deciphered_text_test)

############################

text_test_part = text_test[0:100]
enciphered_text_test = encrypt(text_test_part,key)
print(key)
truekey = key


n_koef = int(input("enter the n-gramm dimension: "))
dict_frequency_training = get_n_gramm(n_koef, text_training_part)
min_p = np.min(list(dict_frequency_training.values()))
dict_to_file(dict_frequency_training,"test_dict_freq.txt")

N = int(input("enter the number of repetitions: "))
identity_score_array = np.zeros(N)
frequency_score_array = np.zeros(N)
hamming_distance_array = np.zeros(N)
marcov_chain_score_array = np.zeros(N)
max_score = 0


score_num, max_score = select_metric(max_score)


max_key = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
random.shuffle(max_key)
size = 1400


koef = 1
a = -1
b = a+koef

if (score_num == 1 and n_koef == 2):
    max_key = ' ETAOINSHRDLCUMWFGYPBVKJXQZ'
    for i in range(N):
        ### hill climbing
        # здесь меняем ключ попарно
        if (a+koef < 26):
            a += 1
            b = a + koef
        else:
            a = 0
            b = a + koef
            koef += 1

        key = swap(max_key, a, b)

        deciphered_text_test = decrypt(enciphered_text_test, key)
        dict_frequency_test = get_n_gramm(n_koef, deciphered_text_test)

        score = use_score(score_num, dict_frequency_test, dict_frequency_training, deciphered_text_test, key, truekey, text_test_part, n_koef)
        if (score < max_score):
            max_score = score
            max_key = copy.copy(key)


        identity_score_array[i] = identityscore(truekey, max_key)


        deciphered_text_test = decrypt(enciphered_text_test, max_key)

        dict_frequency_test = get_n_gramm(n_koef, deciphered_text_test)
        frequency_score_array[i] = frequency_ngram_score(dict_frequency_test, dict_frequency_training)
        hamming_distance_array[i] = hamming_distance(deciphered_text_test, text_test_part)
        marcov_chain_score_array[i] = marcov_chain_score(deciphered_text_test, dict_frequency_training, n_koef)
        if (a == 0 and b == 26):
            koef = 1
            a = -1
            b = a + koef
else:
    for i in range (N):
        ### hill climbing
        # здесь меняем ключ попарно
        a = random.randint(0, 26)
        b = random.randint(0, 26)
        while (b == a):
            b = random.randint(0, 26)
        key = swap(max_key, a, b)

        deciphered_text_test = decrypt(enciphered_text_test, key)
        dict_frequency_test = get_n_gramm(n_koef, deciphered_text_test)

        score = use_score(score_num, dict_frequency_test, dict_frequency_training, deciphered_text_test, key, truekey,
                          text_test_part, n_koef)
        if (score_num > 2 and score > max_score) or (score_num < 3 and score < max_score):
            max_score = score
            max_key = copy.copy(key)

        identity_score_array[i] = identityscore(truekey, max_key)
        deciphered_text_test = decrypt(enciphered_text_test, max_key)

        dict_frequency_test = get_n_gramm(n_koef, deciphered_text_test)
        frequency_score_array[i] = frequency_ngram_score(dict_frequency_test, dict_frequency_training)
        hamming_distance_array[i] = hamming_distance(deciphered_text_test, text_test_part)
        marcov_chain_score_array[i] = marcov_chain_score(deciphered_text_test, dict_frequency_training, n_koef)


print(deciphered_text_test)
print()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

ax1.plot(identity_score_array, label='Identity Score', linewidth=2.0, color='blue')
ax1.set_title("Identity Score")
ax1.grid(True)
ax1.set_xlabel('Index')
ax1.set_ylabel('Score')
ax1.legend()

ax2.plot(frequency_score_array, label='Frequency Score', linewidth=2.0, color='green')
ax2.set_title("Frequency Score")
ax2.grid(True)
ax2.set_xlabel('Index')
ax2.set_ylabel('Score')
ax2.legend()

ax3.plot(hamming_distance_array, label='Hamming Distance', linewidth=2.0, color='red')
ax3.set_title("Hamming Distance")
ax3.grid(True)
ax3.set_xlabel('Index')
ax3.set_ylabel('Distance')
ax3.legend()

ax4.plot(marcov_chain_score_array, label='Markov Chain Score', linewidth=2.0, color='purple')
ax4.set_title("Markov Chain Score")
ax4.grid(True)
ax4.set_xlabel('Index')
ax4.set_ylabel('Score')
ax4.legend()

# Установите автоматическое масштабирование для каждого графика
for ax in [ax1, ax2, ax3, ax4]:
    ax.autoscale(enable=True, axis='both', tight=False)

plt.tight_layout()  # Для лучшего расположения графиков
plt.show()
