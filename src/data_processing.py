import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    data['Annual Salary'] = data['Annual Salary'].str.replace(
        '$', '').str.replace(',', '').str.strip().astype(float)
    return data
