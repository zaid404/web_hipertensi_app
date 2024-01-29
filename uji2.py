import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from io import BytesIO
import base64

class HypertensionDiagnosis2:
    def __init__(self):
        pass

    def run_diagnosis(self):
        test_data = np.genfromtxt('hasil_uji_datatest.csv', delimiter=',', skip_header=1, dtype=str)
        actual_labels = test_data[:, 1]
        diagnosis_result = test_data[:, 2]
        p_x_h_ya_products = test_data[:, 3]
        p_x_h_tidak_products = test_data[:, 4]
        correct_predictions = 0
        total_samples = len(test_data)

        for actual, diagnosis in zip(actual_labels, diagnosis_result):
            if actual == diagnosis:
                correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100
        print("\nAccuracy: {:.2f}%".format(accuracy))

        actual_labels = [1 if label == "hypertension" else 0 for label in actual_labels]

        fpr, tpr, thresholds = roc_curve(actual_labels, p_x_h_ya_products.astype(float))
        auc_score = roc_auc_score(actual_labels, p_x_h_ya_products.astype(float))

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        confusion_matrix_image_buffer = BytesIO()
        plt.savefig(confusion_matrix_image_buffer, format="png")
        confusion_matrix_image_buffer.seek(0)

        confusion_matrix_image_base64 = base64.b64encode(confusion_matrix_image_buffer.read()).decode("utf-8")
        confusion_matrix_image_buffer.close()

        return accuracy, confusion_matrix_image_base64

# Create an instance of the HypertensionDiagnosis class
diagnosis = HypertensionDiagnosis2()
# Run the diagnosis
# accuracy, confusion_matrix_image_base64 = diagnosis.run_diagnosis()

# Run the diagnosis
#diagnosis.run_diagnosis()
