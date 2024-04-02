import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('data.csv')
df = df.dropna(subset=['tempo'])

features = df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', NearestNeighbors(n_neighbors=6))
])

pipeline.fit(features)

def get_recommendations(song_id, data, model_pipeline):

    song_index = data.loc[data['id'] == song_id].index[0]
    
    song_features = features.iloc[[song_index]]
    
    _, indices = model_pipeline.named_steps['knn'].kneighbors(model_pipeline.named_steps['scaler'].transform(song_features), n_neighbors=2)
    
    recommended_index = indices[0][1]
    
    return data.iloc[recommended_index]['id']


song_id = df.sample(1)['id'].values[0]
recommendations = get_recommendations(song_id, df, pipeline)

artist, song_name = df.loc[df['id'] == song_id, ['artist', 'name']].values[0]
recom_artist, recom_song_name = df.loc[df['id'] == recommendations, ['artist', 'name']].values[0]

print(f"Based on the song '{song_name}' by {artist}, we recommend '{recom_song_name}' by {recom_artist}.")

joblib.dump(pipeline, 'knn_song_recommender_pipeline.pkl')
