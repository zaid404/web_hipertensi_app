import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load data from prior.csv
data = np.genfromtxt('prior.csv', delimiter=',', dtype=str)

def calculate_p_x_h(attribute):
    index = np.where(data[:, 0] == attribute)[0]
    if len(index) == 0:
        return None, None
    values = data[index[0], 1:].astype(float)
    p_x_h_ya = np.prod(values[::2])  # Multiply every other value starting from index 0
    p_x_h_tidak = np.prod(values[1::2])  # Multiply every other value starting from index 1
    return p_x_h_ya, p_x_h_tidak

# Define the relevant attributes
options = [
    ['Jenis Kelamin', 'Usia', 'Pusing', 'Berat diTengkuk', 'Sesak Nafas', 'Jantung Berdebar',
     'Tekanan Darah Sistolik', 'Tekanan Darah Diastolik'],
    ['Laki-Laki', 'Remaja', 'Ya', 'Ya', 'Ya', 'Ya', '100 mmHg', '60 mmHg'],
    ['Perempuan', 'Dewasa', 'Tidak', 'Tidak', 'Tidak', 'Tidak', '110 mmHg', '70 mmHg'],
    ['', 'Lansia', '', '', '', '', '120 mmHg', '80 mmHg'],
    ['', '', '', '', '', '', '130 mmHg', '90 mmHg'],
    ['', '', '', '', '', '', '140 mmHg', '100 mmHg'],
    ['', '', '', '', '', '', '150 mmHg', ''],
    ['', '', '', '', '', '', '160 mmHg', ''],
    ['', '', '', '', '', '', '170 mmHg', ''],
    ['', '', '', '', '', '', '180 mmHg', ''],
    ['', '', '', '', '', '', '190 mmHg', ''],
    ['', '', '', '', '', '', '200 mmHg', '']
]

# Calculate P(X|H) by multiplying the probabilities
def calculate_probabilities(options):
    probabilities = [calculate_p_x_h(attr) for attr in options]
    p_x_h_ya_product = 1.0
    p_x_h_tidak_product = 1.0
    for i, attr in enumerate(options):
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

def perform_diagnosis(p_x_h_ya_product, p_x_h_tidak_product, p_ya, p_tidak):
    p_ya = float(p_ya)  # Convert p_ya to float
    p_tidak = float(p_tidak)  # Convert p_tidak to float

    p_final_ya = p_x_h_ya_product * p_ya
    p_final_tidak = p_x_h_tidak_product * p_tidak

    print("P(X|Hasil=Ya) * P(Ya) =", p_final_ya)
    print("P(X|Hasil=Tidak) * P(Tidak) =", p_final_tidak)

    if p_final_ya >= p_final_tidak:
        return ("Hasil dari nilai probabilitas akhir terbesar berada di kelas Ya, maka orang tersebut mengalami penyakit hipertensi.")
    else:
        return ("Hasil dari nilai probabilitas akhir terbesar berada di kelas Tidak, maka orang tersebut tidak mengalami penyakit hipertensi.")


@app.route('/')
def index():
    attribute_names = [
        'Jenis Kelamin', 'Usia', 'Pusing', 'Berat diTengkuk', 
        'Sesak Nafas', 'Jantung Berdebar', 'Tekanan Darah Sistolik', 
        'Tekanan Darah Diastolik'
    ]
    
    options = [
        ['Laki-Laki', 'Perempuan'],
        ['Remaja', 'Dewasa', 'Lansia'],
        ['Ya', 'Tidak'],
        ['Ya', 'Tidak'],
        ['Ya', 'Tidak'],
        ['Ya', 'Tidak'],
        ['100 mmHg', '110 mmHg', '120 mmHg', '130 mmHg', '140 mmHg', '150 mmHg', '160 mmHg', '170 mmHg', '180 mmHg', '190 mmHg', '200 mmHg'],
        ['60 mmHg', '70 mmHg', '80 mmHg', '90 mmHg', '100 mmHg']
    ]
    
    return render_template('index.html', attribute_names=attribute_names, options=options)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    selected_options = request.form.getlist('selected_options')
    if not selected_options:
        return redirect(url_for('index'))
    
    p_ya = float(data[0, 1])  # Convert to float
    p_tidak = float(data[0, 2])  # Convert to float
    p_x_h_ya_product, p_x_h_tidak_product = calculate_probabilities(selected_options)
    result = perform_diagnosis(p_x_h_ya_product, p_x_h_tidak_product, p_ya, p_tidak)
    
    formatted_calculations = []
    for i, attr in enumerate(selected_options):
        p_x_h_ya, p_x_h_tidak = calculate_p_x_h(attr)
        formatted_calculations.append({
            'attribute': attr,
            'p_x_h_ya': p_x_h_ya,
            'p_x_h_tidak': p_x_h_tidak
        })
    
    return render_template('result.html', formatted_calculations=formatted_calculations,
                           p_x_h_ya_product=p_x_h_ya_product, p_x_h_tidak_product=p_x_h_tidak_product,
                           p_final_ya=p_x_h_ya_product * p_ya, p_final_tidak=p_x_h_tidak_product * p_tidak,
                           diagnosis_result=result)


if __name__ == '__main__':
    app.run(debug=True)
