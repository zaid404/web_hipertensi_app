import base64
from io import BytesIO
import numpy as np
from flask import Flask, render_template, request, redirect, url_for,flash
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from uji1 import HypertensionDiagnosis1
from uji2 import HypertensionDiagnosis2
from flask import Flask, render_template, request, redirect, url_for, session
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Email
from functools import wraps
import glob
import os
from proses import Parse,Attr,Priory
from proc import Parse as ParseFromProc, Attr as AttrFromProc, Priory as PrioryFromProc
import json
from werkzeug.serving import run_simple
from sklearn.model_selection import train_test_split
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan kunci rahasia Anda sendiri
bcrypt = Bcrypt()
admin_file = "admin.csv"
app.config['disable_print'] = True
#@disable_print



# Load data from prior.csv


def disable_print(func):
    def wrapper(*args, **kwargs):
        # Redirect sys.stdout to a custom stream that does nothing
        class NullStream:
            def write(self, s):
                pass
        
        if app.config['disable_print']:
            sys.stdout = NullStream()
        return func(*args, **kwargs)
    return wrapper
    

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/uji', methods=['GET', 'POST'])
def input_and_uji():
    diagnosis1 = HypertensionDiagnosis1()
    diagnosis2 = HypertensionDiagnosis2()
    precision = None
    recall = None
    specificity = None
    f1 = None
    if request.method == 'POST':
        #banyak = int(request.form['banyak'])
        
        # Baca file data_set.csv
        df = pd.read_csv('data_set.csv')
        df = df.copy()
        # Split dataset into 80% training and 20% testing
        df.dropna(inplace=True)
        df = df.applymap(lambda x: str(x).replace(',', '').replace('"', ''))
        df = df.applymap(lambda x: float(x) if x.isdigit() else x)
        data_set_80, data_set_20 = train_test_split(df, test_size=0.2, random_state=42)
        data_set_80.to_csv('data_set.csv', index=False)
        
        data_set_20.insert(0, 'No', range(1, 1 + len(data_set_20)))
        data_set_20.to_csv('data_set_20.csv', index=False)
        # Menyimpan hasil ke file CSV baru
        

        #print(f"{banyak} data telah diambil secara acak dan disimpan dalam file 'rand_data_set.csv'.")

        # Setelah semua proses selesai, jalankan diagnosis
        #accuracy1, confusion_matrix_image_base641 = diagnosis1.run_diagnosis()
        accuracy1, confusion_matrix_image_base641, precision, recall, specificity, f1 = diagnosis1.run_diagnosis()
        accuracy2, confusion_matrix_image_base642 = diagnosis2.run_diagnosis()
        df.copy().to_csv('data_set.csv', index=False)
        total_samples = df.shape[0]
        total_train = data_set_80.shape[0]
        total_test = data_set_20.shape[0]


    else:
        accuracy1 = None
        accuracy2 = None
        confusion_matrix_image_base641 = None
        confusion_matrix_image_base642 = None
        total_samples = None
        total_train = None
        total_test = None

    return render_template('uji.html',total_samples=total_samples, total_train=total_train, total_test=total_test, accuracy1=accuracy1, confusion_matrix_image_base641=confusion_matrix_image_base641,
                       precision=precision, recall=recall, specificity=specificity, f1=f1,
                       accuracy2=accuracy2, confusion_matrix_image_base642=confusion_matrix_image_base642)


#data = np.genfromtxt('prior.csv', delimiter=',', dtype=str)
def calculate_p_x_h(attribute):
    index = np.where(data[:, 0] == attribute)[0]
    if len(index) == 0:
        return None, None
    values = data[index[0], 1:].astype(float)
    p_x_h_ya = np.prod(values[::2])  # Multiply every other value starting from index 0
    p_x_h_tidak = np.prod(values[1::2])  # Multiply every other value starting from index 1
    return p_x_h_ya, p_x_h_tidak

# Load attributes from attr.csv
attributes_df = pd.read_csv('attr.csv')
attribute_names = list(attributes_df.columns)
# Define the relevant attributes using the data from 'attr.csv'
options = attributes_df.values.tolist()


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


                           

@app.route('/proses_data')
def proses_data():
    process_message = None


    
    files_to_remove = ['attr.csv', 'attribute_names.csv', 'options.csv', 'parse.csv', 'prior.csv']
    for file in files_to_remove:
        if os.path.isfile(file):
            os.remove(file)
            print(f"File {file} telah dihapus.")
        else:
            print(f"File {file} tidak ditemukan.")
    
    try:
        parse = Parse()
        attr_processor = Attr()
        attr_processor.process_data()
        attr_processor.modify_data()
        attr_processor.save_attribute_names()
        data_file = 'parse.csv'
        conditional_file = 'prior.csv'
        priory_processor = Priory(data_file, conditional_file)
        priory_processor.calculate_and_write_conditional_probabilities()
        process_message = 'Data berhasil diproses'
    except Exception as e:
        print(f"Error: {e}")
        process_message = 'Data gagal diproses'

    return render_template('proses_data.html', process_message=process_message)



@app.route('/proses_diskrit')
def proses_data_diskrit():
    process_message = None

    files_to_remove = ['attr.csv', 'attribute_names.csv', 'options.csv', 'parse.csv', 'prior.csv']
    for file in files_to_remove:
        if os.path.isfile(file):
            os.remove(file)
            print(f"File {file} telah dihapus.")
        else:
            print(f"File {file} tidak ditemukan.")
    
    try:
        parse_proc = ParseFromProc()
        attr_proc = AttrFromProc()
        attr_proc.process_data()
        attr_proc.modify_data()
        attr_proc.save_attribute_names()
        data_file = 'parse.csv'
        conditional_file = 'prior.csv'
        priory_proc = PrioryFromProc(data_file, conditional_file)
        priory_proc.calculate_and_write_conditional_probabilities()
        process_message = 'Data berhasil diproses'
    except Exception as e:
        print(f"Error: {e}")
        process_message = 'Data gagal diproses'

    return render_template('proses_data.html', process_message=process_message)



@app.route('/solusiya')
def solusiya():
    with open('solusiya.csv', 'r') as file2:
        data2 = file2.read()

    return render_template('solusiya.html', data2=data2)
    
@app.route('/solusitidak')
def solusitidak():
    with open('solusitidak.csv', 'r') as file2:
        data2 = file2.read()

    return render_template('solusitidak.html', data2=data2)
    
@app.route('/cek-data')
def cek_data():

    with open('prior.csv', 'r') as file2:
        data2 = file2.read()

    return render_template('cek-data.html', data2=data2)

@app.route('/view_data')
def view_data():
    datas = {}    
    dirs = glob.glob("*.csv")    
    for f in dirs:
        datas[f] = get_data_json_file(f)
    return render_template('index_view.html', dirs=dirs, datas=datas)

#@app.route('/cek-data2', methods=['GET'])
@app.route('/cek-data2/<file>')
def cek_data2(file):
    results = []
    with open(file, 'r', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for r in data:
            results.append(r)

    fieldnames = results[0] if results else []

    return render_template('cek-data2.html', results=results, fieldnames=fieldnames, len=len)

def cek_data2():
    results = []

    # Baca file "prior.csv"
    with open('prior.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            results.append(dict(row))

    fieldnames = [key for key in results[0].keys()]

    return render_template('cek-data2.html', results=results, fieldnames=fieldnames, len=len)


@app.route('/diagnosa')
def diagnosa():

    attribute_names = pd.read_csv('attribute_names.csv').columns.tolist()
    options_df = pd.read_csv('options.csv')
    options = options_df.values.tolist()


    #options = [[opt for opt in sublist if str(opt) != 'nan'] for sublist in options_df.values.tolist()]
    #options = [[opt for opt in sublist if pd.notna(opt) and opt != ''] for sublist in options_df.values.tolist()]
    options = []
    for sublist in options_df.values.tolist():
        nan_count = 0
        new_sublist = []
        for opt in sublist:
            if pd.isna(opt):
                if nan_count == 0:  # Only add the first NaN value encountered
                    new_sublist.append(opt)
                    nan_count += 1
            else:
                new_sublist.append(opt)
        options.append(new_sublist)







    # Membaca file CSV dengan deskripsi atribut
    attribute_descriptions_df = pd.read_csv('attribute_descriptions.csv')

    # Mengonversi DataFrame menjadi kamus dengan nama atribut sebagai kunci dan deskripsi sebagai nilai
    attribute_descriptions = dict(zip(attribute_descriptions_df['Attribute'], attribute_descriptions_df['Description']))
    return render_template('index.html', attribute_names=attribute_names, options=options, attribute_descriptions=attribute_descriptions)

    #return render_template('index.html', attribute_names=attribute_names, options=options)



@app.route('/diagnose', methods=['POST'])
def diagnose():
    global data
    data = np.genfromtxt('prior.csv', delimiter=',', dtype=str)
    selected_options = request.form.getlist('selected_options')
    print(request.form)
    print(selected_options)
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

class AdminForm(FlaskForm):
    nama = StringField('Nama', validators=[DataRequired()])
    no_hp = StringField('No HP', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        admin_data = pd.read_csv(admin_file)

        for index, row in admin_data.iterrows():
            stored_password = row['password'].encode('utf-8')
            if bcrypt.check_password_hash(stored_password, password) and row['email'] == email:
                session['logged_in'] = True
                return redirect(url_for('home'))

        return "Login gagal. Coba lagi."

    return render_template('login.html')
 
# Membaca dataset dari file CSV
dataset = pd.read_csv("data_set.csv")
# Memisahkan kolom-kolom berdasarkan tipe data
numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = dataset.select_dtypes(include=['object']).columns
   
def get_count_plot_html():
    try:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Hypertension', data=dataset)
        plt.title('Count Plot Hypertension')
        plt.xlabel('Hypertension')
        plt.ylabel('Count')

        count_plot_html = plot_to_html()
        plt.close()

        return count_plot_html
    except Exception as e:
        print(f"Error during plot_to_html: {e}")
        return None
def get_histograms_html():
    try:
        histograms_html = []
        for num_column in numerical_columns:
            if num_column != 'Hypertension':
                plt.figure(figsize=(8, 6))
                sns.histplot(dataset[num_column], bins=20, kde=True)
                plt.title(f'Distribusi {num_column}')
                plt.xlabel(num_column)
                plt.ylabel('Frekuensi')

                histogram_html = plot_to_html()
                histograms_html.append({'column': num_column, 'html': histogram_html})

                plt.close()

        return histograms_html
    except Exception as e:
        print(f"Error during plot_to_html: {e}")
        return None

def get_heatmap_html():
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
        plt.title('Heatmap Korelasi Antar Atribut')

        heatmap_html = plot_to_html()
        plt.close()

        return heatmap_html
    except Exception as e:
        print(f"Error during plot_to_html: {e}")
        return None

def get_pie_charts_html():
    try:
        pie_charts_html = []
        for cat_column in categorical_columns:
            if cat_column != 'Hypertension':
                plt.figure(figsize=(6, 6))
                dataset[cat_column].value_counts().plot.pie(autopct='%1.1f%%', labels=dataset[cat_column].unique())
                plt.title(f'Proporsi {cat_column} berdasarkan Hypertension')

                pie_chart_html = plot_to_html()
                pie_charts_html.append({'column': cat_column, 'html': pie_chart_html})

                plt.close()

        return pie_charts_html
    except Exception as e:
        print(f"Error during plot_to_html: {e}")
        return None

def plot_to_html():
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_html = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_html
@app.route('/chart')    
def chart():
    # Count Plot untuk Kolom Hypertension
    count_plot_html = get_count_plot_html()

    # Histogram untuk Kolom Numerik
    histograms_html = get_histograms_html()

    # Heatmap untuk Korelasi Antar Atribut
    heatmap_html = get_heatmap_html()

    # Pie Chart untuk Kolom Kategorikal
    pie_charts_html = get_pie_charts_html()

    # Render halaman HTML dengan grafik
    return render_template('chart.html', count_plot_html=count_plot_html,
                           histograms_html=histograms_html, heatmap_html=heatmap_html,
                           pie_charts_html=pie_charts_html)

@app.route('/admin_fuc')
def admin_fuc():


    # Render halaman HTML dengan grafik
    return render_template('admin_fuc.html')
    
@app.route('/back')
def back():
    # mengarahkan pengguna kembali ke halaman sebelumnya
    return redirect(url_for('home'))

@app.route('/admin')
@login_required
def admin_list():
    results = []

    # Baca file "prior.csv"
    with open('admin.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            results.append(dict(row))

    fieldnames = [key for key in results[0].keys()]

    return render_template('cek-data2.html', results=results, fieldnames=fieldnames, len=len)

@app.route('/add_admin_form', methods=['GET', 'POST'])
@login_required
def add_admin_form():
    form = AdminForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            nama = form.nama.data
            no_hp = form.no_hp.data
            email = form.email.data
            password = form.password.data

            # Enkripsi password
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

            # Tambahkan data admin baru ke berkas CSV
            new_admin = {"nama": nama, "no_hp": no_hp, "email": email, "password": hashed_password}
            admin_data = pd.read_csv(admin_file)

            new_admin_df = pd.DataFrame([new_admin])
            admin_data = pd.concat([admin_data, new_admin_df], ignore_index=True, sort=False)
            admin_data.to_csv(admin_file, index=False)

            return redirect(url_for('cek_data2', file='admin.csv'))


    return render_template('add_admin_form.html', form=form)

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

def get_json_file(file):
    return f"{file}.json"

def get_data_json_file(file):
    filename_json = get_json_file(file)
    print(filename_json)
    if os.path.isfile(filename_json):
        with open(filename_json, "r") as f:
            data = json.load(f)
            # print(data)
            return data
    return None

@app.route('/savedescription/<file>')
def save_description(file):
    with open(get_json_file(file), "w") as f:
        json.dump(request.args.to_dict(), f)

    return redirect("/edit-data")

@app.route('/updatesummary/<file>', methods=["POST"])
def update_summary(file):
    print(request.form.to_dict())
    filename = get_json_file(file)
    if os.path.exists(filename) is False:
        with open(filename, "w") as f:
            json.dump({"sumary": request.data.decode('utf-8')}, f)
    else:
        data = get_data_json_file(file)
        with open(filename, "w") as f:
            data["sumary"] = request.data.decode('utf-8')
            json.dump(data, f)
    return ""

@app.route('/editdescription/<file>')
def edit_description(file):
    ar = request.args
    if len(ar) != 0:
        print(ar)
    # data = json.loads(request.args['data'])
    data = get_data_json_file(file)
    return render_template('editdescription.html', file=file, data=data)

@app.route('/save/<file>', methods=["POST"])
def save(file):
    data = json.loads(request.form['data'])
    with open(file, 'w', newline='') as f:
        wr = csv.writer(f)
        for item in data:
            wr.writerow(item)
    return Response(status=200)

@app.route('/edit/<file>')
def edit(file):
    result = []
    description = get_data_json_file(file)
    with open(file, 'r', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for r in data:
            result.append(r)
    return render_template('edit.html', file=file, data=result, description=description)

@app.route('/edit-data')
def hello():
    datas = {}    
    dirs = glob.glob("*.csv")    
    for f in dirs:
        datas[f] = get_data_json_file(f)
    return render_template('index_edit.html', dirs=dirs, datas=datas)



app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB dalam byte
UPLOAD_FOLDER = os.path.basename('/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    upload_message = None  # upload 

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = 'data_set.csv'
            f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(f)
            if os.path.exists(f):  # 
                upload_message = f'data {file.filename} berhasil di upload '# {filename}'
            else:
                upload_message = 'Upload gagal'

    return render_template('upload.html', upload_message=upload_message, title='Upload File')





if __name__ == '__main__':
    #app.run(host='0.0.0.0')
    #app.run(host='127.0.0.1', port=5000, debug=True)
    #run_simple('127.0.0.1', 5000, app, use_reloader=True)
    #run_simple('0.0.0.0', 5000, app, use_reloader=True)
    run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True)
