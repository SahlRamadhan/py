import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Membaca data
df = pd.read_excel('Latihan Bayes.xlsx')
print(df)

# Transformasi label menjadi angka
ubahAngka = LabelEncoder()
df['JURUSAN'] = ubahAngka.fit_transform(df['JURUSAN'])
df['GENDER'] = ubahAngka.fit_transform(df['GENDER'])
df['ASAL_SEKOLAH'] = ubahAngka.fit_transform(df['ASAL_SEKOLAH'])
df['RERATA_SKS'] = ubahAngka.fit_transform(df['RERATA_SKS'])
df['ASISTEN'] = ubahAngka.fit_transform(df['ASISTEN'])
df['STUDY'] = ubahAngka.fit_transform(df['STUDY'])
print(df)

# Memisahkan atribut dan class label
X = df.iloc[:, 0:5].values
y = df.iloc[:, 5].values  # Mengambil kolom ke-5 sebagai class label
print('nilai atribut yang bukan class label \n', X)
print('nilai atribut class label \n', y)

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Melatih model Naive Bayes
naive_bayes = MultinomialNB()
latih_model = naive_bayes.fit(X_train, y_train)

# Melakukan prediksi pada data uji
prediksi = latih_model.predict(X_test)
print('hasil prediksi : ', prediksi)

# Memilih satu data untuk prediksi tunggal
soal = [[1, 0, 0, 0, 1]]
prediksi_single = latih_model.predict(soal)
print('hasil prediksi untuk satu data: ', prediksi_single)

# Peluang untuk data tunggal
peluang = latih_model.predict_proba(soal)
print('peluang : ', peluang)

print(f'Prediksi untuk input tersebut adalah: {peluang[0]}')

# Melakukan inverse transform untuk mendapatkan label asli
predicted_label = ubahAngka.inverse_transform(np.argmax(peluang, axis=1))
print(predicted_label)

# Menghitung metrik untuk data uji
accuracy = accuracy_score(y_test, prediksi)
precision = precision_score(y_test, prediksi, average='weighted')  # Menambahkan 'average' jika lebih dari dua kelas
recall = recall_score(y_test, prediksi, average='weighted')  # Sama dengan precision
f1 = f1_score(y_test, prediksi, average='weighted')  # Sama dengan precision dan recall

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

class_report = classification_report(y_test, prediksi, target_names=ubahAngka.classes_)
print("Classification Report:\n", class_report)

# Membuat confusion matrix
cm = confusion_matrix(y_test, prediksi)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted') 
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()