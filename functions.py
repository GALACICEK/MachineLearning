
import pandas as pd
import numpy as np
import matplotlib as plt
import statsmodels as sm


#düzenle
def backward_elimination(data, y, significance_level=0.5):
    # Bağımlı değişkeni belirle
    #y = data.iloc[:, -1]
    
    # Bağımsız değişkenlerin matrisini oluştur
    X = np.append(arr=np.ones((len(data), 1)).astype(int), values=data.iloc[:, :-1], axis=1)

    # Modeli oluştur ve p değerlerini al
    model = sm.OLS(y, X).fit()
    p_values = model.pvalues

    # Significance level'dan büyük p değerlerine sahip sütunları belirle
    columns_to_drop = [col for col in data.columns if p_values[col] > significance_level]

    # Seçilen sütünları elimine et
    data = data.drop(columns=columns_to_drop)

    return data
