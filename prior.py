import pandas as pd
import os
import numpy as np
import csv

# Memuat dataset
data = pd.read_csv("dataset.csv")
"""## mencari probabilitas hipotesis P(H) untuk masing masing Kelas
p( x) = x/total data(ya|tidak)

"""
data = pd.read_csv("dataset.csv")
array_data = data.values
nilai_hipertensi_kasus = array_data[0, 2]
nilai_hipertensi_tidak = array_data[0, 4]
nilai_hipertensi_ya = array_data[0, 3]
p_ya = nilai_hipertensi_ya / nilai_hipertensi_kasus
p_tidak = nilai_hipertensi_tidak / nilai_hipertensi_kasus
file_path = 'prior.csv'
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    row = ["p(x)", p_ya, p_tidak]  # Changed parentheses to square brackets
    writer.writerow(row)
"""## langkah selanjutnya adalah menghitung probabilitas kondisi tertentu
(probabilitas X) berdasarkan probabilitas tiap hipotesis(probabilitas H) atau dinamakan probabilitas prior P(X|H)
P(X|H)
Ya Tidak
"""
file_path = 'prior.csv'
def calculate_p_x_h(v, w, x, y):
    p_x_h_ya = array_data[v, w] / nilai_hipertensi_ya
    p_x_h_tidak = array_data[x, y] / nilai_hipertensi_tidak
    return p_x_h_ya, p_x_h_tidak

with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    for i in range(1, 33):
        p_x_h_ya, p_x_h_tidak = calculate_p_x_h(i, 3, i, 4)
        #row = [array_data[i, 0], array_data[i, 1], f"P(X|H) Ya: {p_x_h_ya:.2f}", f"P(X|H) Tidak: {p_x_h_tidak:.2f}"]
        row = [array_data[i, 0] + " " + array_data[i, 1], p_x_h_ya,p_x_h_tidak]
        writer.writerow(row)
        print(array_data[i, 0] + " " + array_data[i, 1], f"P(X|H) Ya: {p_x_h_ya:.2f}", f"P(X|H) Tidak: {p_x_h_tidak:.2f}")

