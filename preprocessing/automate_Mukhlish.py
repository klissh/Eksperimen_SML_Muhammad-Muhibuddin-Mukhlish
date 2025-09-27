import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def automate_preprocessing(raw_data_path, output_folder):
    """
    Memuat data mentah, melakukan scaling pada kolom 'Time' dan 'Amount',
    dan menyimpan hasilnya sebagai file CSV baru.
    """
    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    print("Preprocessing data...")
    # Inisialisasi scaler
    scaler = StandardScaler()

    # Terapkan scaling
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Hapus kolom asli
    df_processed = df.drop(['Time', 'Amount'], axis=1)

    # Pastikan folder output ada
    os.makedirs(output_folder, exist_ok=True)
    
    # Tentukan path file output
    output_path = os.path.join(output_folder, 'creditcard_processed.csv')
    
    # Simpan dataframe yang sudah diproses
    df_processed.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Processed data saved to {output_path}")

if __name__ == '__main__':
    # Definisikan path relatif
    RAW_DATA_PATH = 'dataset_raw/creditcard.csv'
    OUTPUT_FOLDER = 'dataset_preprocessing'
    
    # Jalankan fungsi
    automate_preprocessing(RAW_DATA_PATH, OUTPUT_FOLDER)