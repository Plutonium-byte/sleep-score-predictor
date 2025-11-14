import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline

def prepare_data():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['sleep_disorder'] = df['sleep_disorder'].fillna('None')

    df[['systolic_bp', 'diastolic_bp']] = df['blood_pressure'].str.split('/', expand=True).astype(int)
    df.index = df['person_id']

    del df['person_id']
    del df['blood_pressure']

    return df


# In[37]:
def train_model(df):

    pipeline = make_pipeline(
        DictVectorizer(),
        DecisionTreeRegressor(max_depth=10)
    )

    train_dicts = df.drop(columns='quality_of_sleep').to_dict(orient='records')
    y_train = df['quality_of_sleep'].values

    pipeline.fit(train_dicts, y_train)

    return pipeline

def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f'Model saved to {filename}')

df = prepare_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)

