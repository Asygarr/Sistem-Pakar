from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Mentrain model SVM
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

# Memuat halaman upload file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        # Menyimpan file 
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        # Mengambil data dari file yang sudah diupload
        df = pd.read_excel(file_path)

        # Proses prediksi
        X_input = df.iloc[:, 1:9]
        predictions = clf.predict(X_input)

        # Membuat dataframe baru untuk menyimpan hasil prediksi
        result_df = pd.DataFrame({'Nama': df.iloc[:, 0], 'Hasil': predictions})
        result_df['Hasil'] = result_df['Hasil'].replace({1: 'positif', 0: 'negatif'})

        # Menyimpan hasil prediksi ke dalam file excel
        result_path = 'uploads/hasil_diagnosa.xlsx'
        result_df.to_excel(result_path, index=False)

        # Mengembalikan halaman utama dengan pesan sukses dan link untuk mengunduh hasil prediksi
        return render_template('index.html', message='File uploaded and processed successfully. Download your results below.', result_path=result_path)

# Mengunduh file hasil prediksi
@app.route('/download/<filename>')
def download_file(filename):
    return send_file('uploads/' + filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
