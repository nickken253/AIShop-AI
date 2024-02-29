from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os 
import tensorflow as tf
from keras import Model
import requests
from io import BytesIO
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import linear_kernel
import warnings
import tempfile
warnings.filterwarnings("ignore")

app = Flask(__name__)

model = tf.keras.models.load_model('src/AIShop_model.h5')

df_embeddings = pd.read_csv('src/df_embeddings.csv')

def download_image(url):
    response = requests.get(url)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(response.content)
    temp_file.seek(0)
    return temp_file.name

def predict_embeddings(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    sample_image = model.predict(img)
    return sample_image.flatten()

def get_similarity(img_path):
    sample_image = predict_embeddings(img_path)
    sample_image = pd.DataFrame(sample_image.squeeze())
    sample_similarity = linear_kernel(sample_image.values.reshape(1, -1), df_embeddings)
    return sample_similarity

def normalize_sim(similarity):
    x_min = similarity.min(axis=1)
    x_max = similarity.max(axis=1)
    norm = (similarity-x_min)/(x_max-x_min)[:, np.newaxis]
    return norm

def get_recommendations(df, similarity, n=5):
    sim_scores = list(enumerate(similarity[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:n]
    cloth_indices = [i[0] for i in sim_scores]
    return df['id'].iloc[cloth_indices]

def print_recommendations(img_url):
    img_path = download_image(img_url)
    similarity = get_similarity(img_path)
    norm_similarity = normalize_sim(similarity)
    df = pd.read_csv('src/styles.csv', low_memory=False)
    df['image'] = df['id'].astype(str) + '.jpg'
    recommendation = get_recommendations(df, norm_similarity)
    return recommendation.tolist()

@app.route('/recommendations', methods=['POST'])
def recommend():
    img_url = request.json['img_url']
    recommendation = print_recommendations(img_url)
    return jsonify(recommendation)

if __name__ == '__main__':
    app.run(debug=True, host='10.1.0.4', port=3033)