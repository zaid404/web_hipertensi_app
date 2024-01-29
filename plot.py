import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset dari file CSV
dataset = pd.read_csv("data_set.csv")

# Pie Chart untuk Proporsi Hipertensi
plt.figure(figsize=(6, 6))
dataset['Hypertension'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'Hypertension'])
plt.title('Proporsi Normal dan Hypertension')
plt.show()

# Bar Chart untuk Atribut Kategorikal
categorical_attributes = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
for attribute in categorical_attributes:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=attribute, hue='Hypertension', data=dataset)
    plt.title(f'Frekuensi {attribute} berdasarkan Hypertension')
    plt.xlabel(attribute)
    plt.ylabel('Frekuensi')
    plt.show()

# Box Plot untuk Atribut Numerik
numerical_attributes = ['age_years', 'bmi', 'ap_hi', 'ap_lo']
for attribute in numerical_attributes:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Hypertension', y=attribute, data=dataset)
    plt.title(f'Perbandingan {attribute} antara Normal dan Hypertension')
    plt.xlabel('Hypertension')
    plt.ylabel(attribute)
    plt.show()

# Count Plot untuk Label 'cardio'
plt.figure(figsize=(8, 6))
sns.countplot(x='cardio', hue='Hypertension', data=dataset)
plt.title('Frekuensi Hipertension berdasarkan Label Cardio')
plt.xlabel('Cardio')
plt.ylabel('Frekuensi')
plt.show()
