# Expert System for Hypertension Classification Using the Naive Bayes Method

![Screenshot 1](https://raw.githubusercontent.com/zaid404/web_hipertensi_app/main/Screenshot_15.png)
![Screenshot 2](https://raw.githubusercontent.com/zaid404/web_hipertensi_app/main/prevhomet_14.png)

## Description
Use python 3.8 and pandas pandas==1.3.5 for mode see requirements.txt
This is a web-based expert system that uses the **Naive Bayes** classification method. The system is built using the **Flask** framework and is designed to automate the learning process based on datasets uploaded by the user.

### Key Features
- **Automated Learning**: The system automatically adjusts its learning model based on the uploaded dataset.
- **Binary Classification**: The last row in the dataset (`Hypertension`) is the classification column, limited to two classes: `Hypertension` and `Normal`.
- **Dataset Flexibility**: Columns before the classification column can be adjusted to fit different types of datasets, as long as they meet the binary classification requirement and follow the same format for the last column.

### Additional Use Cases
This expert system can also be used to classify other binary-class datasets. To use the system with a new dataset, upload the dataset and ensure that the last column follows the same format as described above.

### Additional Information from the Study
Hypertension is often called the "silent killer" because it typically shows no symptoms. This study aims to help reduce the prevalence of hypertension in Indonesia by facilitating early detection and raising public awareness. Using the **Naïve Bayes** method combined with **data discretization** through the **CART (Classification and Regression Trees)** method, the dataset includes 11,627 medical records from 4,434 participants in the *Framingham Heart Study* (FHS), conducted by the *National Institutes of Health*. The system estimates the probability of hypertension occurrence based on input risk factors, achieving an accuracy of 84.28%.

For more details, please refer to our journal publication: [DOI: https://doi.org/10.37396/jsc.v7i1.381](https://doi.org/10.37396/jsc.v7i1.381).

### Journal
A detailed journal explaining this expert system has been published. This project is part of the graduation requirements for the Computer Science department. We hope it is beneficial. Thank you!

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/zaid404/web_hipertensi_app.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd web_hipertensi_app
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python app6.py
   ```
## simply just do ```bash run2.sh```
## Contribution

Contributions in the form of pull requests are highly welcome. Please make sure to `fork` this repository and create a new branch for any feature or fix before submitting a pull request.
