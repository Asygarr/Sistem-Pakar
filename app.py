from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import uuid  # Import the uuid module

app = Flask(__name__)

# Mentrain model dengan SVM
data = pd.read_csv('data/dataset.csv')
X = data[data.columns[:8]]
y = data['Hasil']
x_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
clf = SVC()
clf.fit(x_train, y_train)

# Memuat halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Memuat halaman result
@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        # Memuat random unique identifier
        unique_id = str(uuid.uuid4())
        # Menggabungkan unique identifier dengan nama file yang diupload
        file_path = 'uploads/' + unique_id + '_' + file.filename
        file.save(file_path)

        # Mengambil data dari file yang sudah diupload
        df = pd.read_excel(file_path)

        # Proses prediksi
        X_input = df.iloc[:, 1:9]
        predictions = clf.predict(X_input)

        # Membuat dataframe baru untuk menyimpan hasil prediksi
        result_df = pd.DataFrame({'Nama': df.iloc[:, 0], 'Hasil': predictions})
        result_df['Hasil'] = result_df['Hasil'].replace({1: 'positif', 0: 'negatif'})

        # Membuat file hasil prediksi
        result_unique_id = str(uuid.uuid4())
        result_path = 'result/' + result_unique_id + '_hasil_diagnosa.xlsx'
        result_df.to_excel(result_path, index=False)

        # Membaca file hasil prediksi dan mengonversinya ke HTML
        result_html = result_df.to_html(classes='table table-striped', index=False)

        # Mengembalikan halaman result dengan tabel HTML dan link untuk mengunduh hasil prediksi
        return render_template('result.html', result_html=result_html, result_path=result_path)

# Mengunduh file hasil prediksi
@app.route('/download/<filename>')
def download_file(filename):
    return send_file('result/' + filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
