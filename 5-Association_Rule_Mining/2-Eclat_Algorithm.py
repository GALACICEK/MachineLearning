# Equivalence Class Transformation
# Depth First Search

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loading data sets----------------------------------------------
veriler = pd.read_csv('data/sepet.csv', header = None)

t = []
for i in range(len(veriler)):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])




from collections import defaultdict
from itertools import combinations

# Destek değeri
min_support = 2

# Eclat algoritması
def eclat(transactions, min_support):
    # Öğe kümesi ve destek değerlerini saklamak için bir sözlük oluştur
    itemsets = defaultdict(int)

    # Tüm öğelerin tekli kümelerinin desteklerini hesapla
    for transaction in transactions:
        for item in transaction:
            itemsets[item] += 1

    # Destek değeri geçmeyen öğeleri kaldır
    itemsets = {itemset: support for itemset, support in itemsets.items() if support >= min_support}

    # Eclat algoritmasıyla öğe kümelerini oluştur
    for k in range(2, len(itemsets)+1):
        for itemset in combinations(itemsets.keys(), k):
            support = 0
            for transaction in transactions:
                if set(itemset).issubset(transaction):
                    support += 1
            if support >= min_support:
                itemsets[itemset] = support

    return itemsets

# Eclat algoritmasını kullanarak sık görülen öğe kümelerini bul
frequent_itemsets = eclat(t, min_support)

# Sonuçları yazdır
for itemset, support in frequent_itemsets.items():
    print("Frequent Itemset:", itemset)
    print("Support:", support)











