import csv
import pandas as pd
import numpy as np
import os


                
                

# Membaca data dari file CSV dan nama file output
import pandas as pd

class Parse: 
    def __init__(self):
        # Membaca file CSV
        df = pd.read_csv('data_set.csv')

        # Menghapus baris yang memiliki nilai yang hilang (NaN)
        df.dropna(inplace=True)

        # Menghilangkan koma dan tanda kutip dari semua string
        df = df.applymap(lambda x: str(x).replace(',', '').replace('"', ''))

        # Mengubah semua angka menjadi bentuk desimal
        df = df.applymap(lambda x: float(x) if x.isdigit() else x)
        df.to_csv('data_set.csv', index=False)

        # Load the example data from CSV
        self.example_data = df

        # Define the columns for aggregation (excluding 'Hypertension')
        self.columns_for_aggregation = self.example_data.columns.tolist()[:-1]

        # Create a DataFrame for aggregation
        self.aggregated_data = pd.DataFrame(columns=['Atribut', 'Jumlah Kasus', 'hipertensi Ya', 'hipertensi Tidak'])

        # Calculate total row
        total_row = {
            'Atribut': 'TOTAL',
            'Jumlah Kasus': len(self.example_data),
            'hipertensi Ya': len(self.example_data[self.example_data['Hypertension'] == 'hypertension']),
            'hipertensi Tidak': len(self.example_data[self.example_data['Hypertension'] != 'hypertension'])
        }

        # Append total row to the DataFrame
        self.aggregated_data = self.aggregated_data.append(total_row, ignore_index=True)

        # Iterate through each column for aggregation
        for column in self.columns_for_aggregation:
            attribute_values = self.example_data[column].unique()

            for value in attribute_values:
                formatted_column = column.replace(' ', '_')  # Replace spaces with underscores
                attribute_name = f"{formatted_column} {value}"

                subset = self.example_data[self.example_data[column] == value]
                total_cases = len(subset)
                hypertension_yes = len(subset[subset['Hypertension'] == 'hypertension'])
                hypertension_no = total_cases - hypertension_yes

                self.aggregated_data = self.aggregated_data.append({
                    'Atribut': attribute_name,
                    'Jumlah Kasus': total_cases,
                    'hipertensi Ya': hypertension_yes,
                    'hipertensi Tidak': hypertension_no
                }, ignore_index=True)

        # Save aggregated data to CSV
        self.aggregated_data.to_csv('parse.csv', index=False)


# Create an instance of the Parse class and run the aggregation


class Attr:
    def __init__(self):
        pass

    def process_data(self):
        # Load the aggregated data from CSV
        prior_data = pd.read_csv('data_set.csv')

        # Loop through columns and collect unique values for each attribute
        unique_attributes = {}

        # Find the maximum length of unique values
        max_length = 0
        for column in prior_data.columns:
            if column != 'Attribute':
                unique_values = prior_data[column].unique()
                unique_attributes[column] = unique_values
                max_length = max(max_length, len(unique_values))

        # Ensure all arrays have the same length
        for column in unique_attributes:
            while len(unique_attributes[column]) < max_length:
                unique_attributes[column] = list(unique_attributes[column]) + ['']

        # Create a DataFrame to store the unique attributes
        attributes_df = pd.DataFrame(unique_attributes)

        # Save the DataFrame to 'attr.csv' without an index
        attributes_df.to_csv('attr.csv', index=False)

    def modify_data(self):
        # Membaca data dari 'attr.csv'
        data = []
        with open('attr.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the first row
            for row in csvreader:
                # Mengabaikan kolom terakhir dengan mengambil semua elemen kecuali yang terakhir
                modified_row = row[:-1]
                data.append(modified_row)

        # Mentransposisi data
        transposed_data = np.array(data).T.tolist()

        # Menambahkan baris pertama dengan nilai null
        transposed_data.insert(0, [None] * len(transposed_data[0]))

        # Menyimpan data yang sudah dimodifikasi ke dalam 'options.csv'
        with open('options.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(transposed_data)

    def save_attribute_names(self):
        # Read 'attr.csv' and get the first row for attribute names
        with open('attr.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            attribute_names = next(csvreader)

        # Remove the last column
        attribute_names = attribute_names[:-1]

        # Save attribute names to 'attribute_names.csv'
        with open('attribute_names.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(attribute_names)
class Priory:
    def __init__(self, data_file, conditional_file):
        self.data_file = data_file
        self.conditional_file = conditional_file

    def calculate_p_x_h(self, v, w, x, y):
        # Load the aggregated data from CSV
        prior_data = pd.read_csv(self.data_file)
        array_data = prior_data.values

        p_x_h_ya = array_data[v, w] / array_data[0, 2]  # 'hipertensi Ya' count divided by total Ya count
        p_x_h_tidak = array_data[x, y] / array_data[0, 3]  # 'hipertensi Tidak' count divided by total Tidak count
        return p_x_h_ya, p_x_h_tidak

    def calculate_and_write_conditional_probabilities(self):
        if os.path.exists(self.conditional_file):
            os.remove(self.conditional_file)
            print(f"File {self.conditional_file} telah dihapus.")
            
        # Load the aggregated data from CSV
        prior_data = pd.read_csv(self.data_file)
        array_data = prior_data.values

        nilai_hipertensi_kasus = array_data[0, 1]  # Total cases
        nilai_hipertensi_ya = array_data[0, 2]
        nilai_hipertensi_tidak = array_data[0, 3]
        p_ya = nilai_hipertensi_ya / nilai_hipertensi_kasus
        p_tidak = nilai_hipertensi_tidak / nilai_hipertensi_kasus

        with open(self.conditional_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Calculate and write for 'TOTAL' row
            total_p_x_h_ya, total_p_x_h_tidak = self.calculate_p_x_h(0, 2, 0, 3)
            total_row = ['p(x)', p_ya, p_tidak]
            writer.writerow(total_row)

            for i in range(1, len(array_data)):
                p_x_h_ya, p_x_h_tidak = self.calculate_p_x_h(i, 2, i, 3)
                row = [array_data[i, 0], p_x_h_ya, p_x_h_tidak]
                writer.writerow(row)
                #print(array_data[i, 0], f"P(X|H) Ya: {p_x_h_ya:.2f}", f"P(X|H) Tidak: {p_x_h_tidak:.2f}")
                
            formatted_p_x_h_ya = f"{p_x_h_ya:.2f}"
            formatted_p_x_h_tidak = f"{p_x_h_tidak:.2f}"
            print(f"{array_data[i, 0]} P(X|H) Ya: {formatted_p_x_h_ya} P(X|H) Tidak: {formatted_p_x_h_tidak}")

"""

parse = Parse()
attr_processor = Attr()
attr_processor.process_data()
attr_processor.modify_data()
attr_processor.save_attribute_names()
data_file = 'parse.csv'
conditional_file = 'prior.csv'
priory_processor = Priory(data_file, conditional_file)
priory_processor.calculate_and_write_conditional_probabilities()
"""