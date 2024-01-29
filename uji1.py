import base64
from io import BytesIO
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import pandas as pd
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class HypertensionDiagnosis1:
  def __init__(self):
      # Mount the Google Drive to access the data file
     # drive.mount("/content/drive", force_remount=True)
      # Copy the data file from Google Drive to the local Colab environment
      #!cp "/content/drive/MyDrive/collab/prior.csv" "./prior.csv"
      # Load data from prior.csv
      self.data = np.genfromtxt('prior.csv', delimiter=',', dtype=str)
      print(self.data[:30])


      # Define the relevant attributes

      attributes_df = pd.read_csv('attr.csv')
      #attribute_names = list(attributes_df.columns)
      # Define the relevant attributes using the data from 'attr.csv'
      #self.options = attributes_df.values.tolist()
      self.options = [attributes_df.columns.tolist()] + attributes_df.values.tolist()
      # Define the relevant attributes using the data from 'attr.csv'




  def calculate_p_x_h(self, attribute):
      index = np.where(self.data[:, 0] == attribute)[0]
      if len(index) == 0:
          return None, None
      values = self.data[index[0], 1:].astype(float)
      p_x_h_ya = np.prod(values[::2])  # Multiply every other value starting from index 0
      p_x_h_tidak = np.prod(values[1::2])  # Multiply every other value starting from index 1
      return p_x_h_ya, p_x_h_tidak

  def calculate_probabilities(self, selected_options):
      probabilities = [self.calculate_p_x_h(attr) for attr in selected_options]
      p_x_h_ya_product = 1.0
      p_x_h_tidak_product = 1.0
      for i, attr in enumerate(selected_options):
          p_x_h_ya, p_x_h_tidak = probabilities[i]
          if p_x_h_ya is None or p_x_h_tidak is None:
              print(f"Data for attribute '{attr}' is not found.")
          else:
              p_x_h_ya_product *= p_x_h_ya
              p_x_h_tidak_product *= p_x_h_tidak
              print(attr, f"P(X|H) Ya: {p_x_h_ya:.9f}", f"P(X|H) Tidak: {p_x_h_tidak:.9f}")
      print("p_x_h_ya =", p_x_h_ya_product)
      print("p_x_h_tidak =", p_x_h_tidak_product)
      return p_x_h_ya_product, p_x_h_tidak_product

  def diagnose(self, p_x_h_ya_product, p_x_h_tidak_product, p_ya, p_tidak):
      p_final_ya = p_x_h_ya_product * p_ya
      p_final_tidak = p_x_h_tidak_product * p_tidak

      print("P(X|Hasil=Ya) * P(Ya) =", p_final_ya)
      print("P(X|Hasil=Tidak) * P(Tidak) =", p_final_tidak)

      if p_final_ya >= p_final_tidak:
          print("Hasil dari nilai probabilitas akhir terbesar berada di kelas Ya, maka orang tersebut mengalami penyakit hipertensi.")
      else:
          print("Hasil dari nilai probabilitas akhir terbesar berada di kelas Tidak, maka orang tersebut tidak mengalami penyakit hipertensi.")

  def run_diagnosis(self):
      # Load data from datatest.csv
      test_data = np.genfromtxt('data_set_20.csv', delimiter=',', dtype=str, skip_header=1)

      correct_predictions = 0
      total_samples = len(test_data)
      actual_labels = []
      predicted_labels = []

      with open('hasil_uji_datatest.csv', 'w', newline='') as output_file:
          writer = csv.writer(output_file)
          writer.writerow(["No.", "Data Hasil", "Diagnosis Result", "p_x_h_ya_product", "p_x_h_tidak_product"])

          for line in test_data:
              # Get the attribute names
              attribute_names = self.options[0]  # Exclude the first and last elements, which are 'No.' and 'Hasil'

              # Array to store user-selected options (using the attribute data directly from the line)
              selected_options = [f"{attr} {line[i+1]}" for i, attr in enumerate(attribute_names)]

              # Calculate probabilities
              p_ya = float(self.data[0, 1])  # Convert to float
              p_tidak = float(self.data[0, 2])  # Convert to float
              p_x_h_ya_product, p_x_h_tidak_product = self.calculate_probabilities(selected_options)

              # Perform diagnosis
              diagnosis_result = "hypertension" if p_x_h_ya_product >= p_x_h_tidak_product else "Normal"
              actual_label = line[-1]
              actual_labels.append(actual_label)
              predicted_labels.append(diagnosis_result)

              # Write the diagnosis result to the output file
              writer.writerow([line[0], line[-1], diagnosis_result,p_x_h_ya_product, p_x_h_tidak_product])

              # Calculate accuracy
              if line[-1] == diagnosis_result:
                  correct_predictions += 1

              print("\nDiagnosis Result for Line:", line)
              print("Hasil Diagnosis:", diagnosis_result)
              print("------------")

      # Calculate accuracy percentage
      accuracy = (correct_predictions / total_samples) * 100
      print("\nAccuracy: {:.2f}%".format(accuracy))

      # Create a DataFrame for the confusion matrix
      confusion_matrix_df = pd.crosstab(pd.Series(actual_labels, name='Actual'),
                                        pd.Series(predicted_labels, name='Predicted'))
    # Calculate additional evaluation metrics
      true_positive = confusion_matrix_df.loc['hypertension', 'hypertension']
      true_negative = confusion_matrix_df.loc['Normal', 'Normal']
      false_positive = confusion_matrix_df.loc['Normal', 'hypertension']
      false_negative = confusion_matrix_df.loc['hypertension', 'Normal']

# Calculate metrics
      #accuracy2 = accuracy_score(actual_labels, predicted_labels)
      precision = precision_score(actual_labels, predicted_labels, pos_label='hypertension')
      recall = recall_score(actual_labels, predicted_labels, pos_label='hypertension')
      specificity = true_negative / (true_negative + false_positive)
      f1 = f1_score(actual_labels, predicted_labels, pos_label='hypertension')
      print("\nAdditional Evaluation Metrics:")
      print("Accuracy: {:.2f}%".format(accuracy))
      print("Precision (Positive Predictive Value): {:.2f}".format(precision))
      print("Recall (Sensitivity or True Positive Rate): {:.2f}".format(recall))
      print("Specificity (True Negative Rate): {:.2f}".format(specificity))
      print("F1 Score: {:.2f}".format(f1))
      # Plot the confusion matrix as a heatmap
      plt.figure(figsize=(8, 6))
      sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
      plt.xlabel('Predicted Label')
      plt.ylabel('True Label')
      plt.title('Confusion Matrix')
      plt.show()

      confusion_matrix_image_buffer = BytesIO()
      plt.savefig(confusion_matrix_image_buffer, format="png")
      confusion_matrix_image_buffer.seek(0)

      confusion_matrix_image_base64 = base64.b64encode(confusion_matrix_image_buffer.read()).decode("utf-8")
      confusion_matrix_image_buffer.close()

      return accuracy, confusion_matrix_image_base64, precision, recall, specificity, f1
      #,confusion_matrix_image_buffer 

# Create an instance of the HypertensionDiagnosis class
#diagnosis = HypertensionDiagnosis1()
# Run the diagnosis
#diagnosis.run_diagnosis()

