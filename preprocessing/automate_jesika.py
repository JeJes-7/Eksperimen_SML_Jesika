import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data():
    """
    Memuat dataset dari file lokal (localhost).
    Pastikan file 'heart_failure_raw.csv' ada di folder yang sama.
    """
    # Nama file lokal
    filename = 'heart_failure_raw.csv'
    
    # Cek apakah file ada
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            print(f"[INFO] Dataset lokal ditemukan: {filename}")
            print(f"[INFO] Ukuran awal: {df.shape}")
            return df
        except Exception as e:
            print(f"[ERROR] Gagal membaca file: {e}")
            return None
    else:
        # Opsional: Fallback ke URL jika file lokal tidak ketemu (Fitur Smart!)
        print(f"[WARNING] File '{filename}' tidak ditemukan. Mencoba download dari UCI...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
        try:
            df = pd.read_csv(url)
            # Simpan ke lokal biar besok tidak perlu download lagi
            df.to_csv(filename, index=False) 
            print(f"[INFO] Berhasil download dan simpan ke '{filename}'")
            return df
        except Exception as e:
            print(f"[ERROR] Gagal download: {e}")
            return None

def handle_outliers(df, column):
    """
    Menghapus outlier menggunakan metode IQR pada kolom tertentu.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    initial_count = df.shape[0]
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    final_count = df_clean.shape[0]
    
    print(f"[INFO] Outlier handling '{column}': Dibuang {initial_count - final_count} baris.")
    return df_clean.reset_index(drop=True)

def preprocess_data(df):
    """
    Menjalankan seluruh pipeline preprocessing:
    1. Drop kolom 'time'
    2. Binning & Encoding Age
    3. Outlier Removal
    4. Splitting & Scaling
    """
    # 1. Drop 'time' (Data Leakage)
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
    
    # 2. Binning Age
    bins = [0, 60, 75, 120]
    labels = ['Adult', 'Senior', 'Elderly']
    df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    # 3. Encoding (One-Hot Encoding untuk kategori umur)
    df = pd.get_dummies(df, columns=['age_category'], prefix='age')
    
    # 4. Outlier Handling (Hanya pada Serum Creatinine)
    df = handle_outliers(df, 'serum_creatinine')
    
    # 5. Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    
    return X, y

def save_prepared_data(X, y):
    """
    Melakukan Splitting, Scaling, dan Penyimpanan data siap latih.
    """
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling (Standarisasi)
    scaler = StandardScaler()
    
    # Fit pada training, transform pada training & test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Kembalikan ke format DataFrame agar nama kolom tidak hilang
    columns = X.columns
    X_train_final = pd.DataFrame(X_train_scaled, columns=columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=columns)
    
    # Gabungkan kembali dengan target untuk disimpan jadi CSV
    train_set = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
    test_set = pd.concat([X_test_final, y_test.reset_index(drop=True)], axis=1)
    
    # Simpan ke file CSV
    train_set.to_csv('data_train.csv', index=False)
    test_set.to_csv('data_test.csv', index=False)
    
    print("[INFO] Data berhasil disimpan: 'data_train.csv' dan 'data_test.csv'")
    print(f"[INFO] Ukuran Data Train: {train_set.shape}")
    print(f"[INFO] Ukuran Data Test : {test_set.shape}")

if __name__ == "__main__":
    print("--- MULAI PROSES OTOMATISASI ---")
    df = load_data()
    
    if df is not None:
        X, y = preprocess_data(df)
        save_prepared_data(X, y)
    
    print("--- SELESAI ---")
