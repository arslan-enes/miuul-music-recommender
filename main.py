import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(layout = "wide", page_title="Miuul Music Club", page_icon="🎷")

@st.cache_data
def get_data():
    dataframe = pd.read_csv('data.csv')
    return dataframe

@st.cache_data
def get_pipeline():
    pipeline = joblib.load('knn_song_recommender_pipeline.pkl')
    return pipeline

st.title("🎷 :rainbow[Miuul Music Club] 🎷")

main_tab, random_tab, recommendation_tab = st.tabs(["Ana Sayfa", "Rastgele Şarkılar", "Öneri Sistemi"])

# Ana Sayfa

left_col, right_col = main_tab.columns(2) 

left_col.write("""Spotify, 2008 yılında İsveç'te kurulan ve dünya genelinde milyonlarca kullanıcıya sahip bir müzik, podcast ve video akışı sağlayan bir platformdur. Kullanıcılara, çeşitli sanatçıların müziklerine ve podcastlerine erişim imkanı sunar. Ücretsiz ve premium olmak üzere iki farklı abonelik seçeneği bulunur. Ücretsiz versiyon, reklamlarla desteklenir ve bazı sınırlamalar içerirken, premium abonelik aylık bir ücret karşılığında reklamsız dinleme, yüksek kaliteli ses ve çevrimdışı dinleme gibi özellikler sunar.
\nSpotify'ın en dikkat çekici özeldliklerinden biri, kullanıcıların müzik zevklerine uygun yeni şarkılar ve sanatçılar keşfetmelerine yardımcı olan kişiselleştirilmiş çalma listeleri ve öneri sistemidir. Ayrıca, kullanıcılar kendi çalma listelerini oluşturabilir, paylaşabilir ve diğer kullanıcıların çalma listelerine erişebilir. 
\nSpotify, kullanıcı deneyimini zenginleştirmek için sürekli olarak yenilikler yapmaya ve platformunu geliştirmeye odaklanmıştır. Bu yenilikler arasında, sanatçıların ve içerik üreticilerinin takipçileriyle daha etkili bir şekilde etkileşime girebilmeleri için geliştirilen sosyal özellikler de bulunur. Örneğin, kullanıcılar en sevdikleri sanatçıların yeni yayınladığı müzikleri anında keşfedebilir, sanatçıların playlistlerini takip edebilir ve arkadaşlarıyla müzik paylaşabilir.""")

right_col.image("spoti.jpg")

# Rastgele

df = get_data()

col1, col2, col3, col4, col5 = random_tab.columns(5, gap="small")
columns = [col1, col2, col3, col4, col5]
empty_col1, empty_col2, empty_col3 = random_tab.columns([4,3,2])

if empty_col2.button("Rastgele Şarkı Öner"):

    random_songs = df[~df["preview_url"].isna()].sample(5)

    for i, col in enumerate(columns):

        col.image(random_songs.iloc[i]['url'])
        col.write(f"**{random_songs.iloc[i]['name']}**")
        col.write(f"*{random_songs.iloc[i]['artist']}*")
        if str(random_songs.iloc[i]['preview_url']) != 'nan':
            col.audio(random_songs.iloc[i]['preview_url'])

# Öneri Sistemi

pipeline = get_pipeline()

col_features1, col_features2, col_recommendation = recommendation_tab.columns(3)


acousticness = col_features1.slider("Akkustiklik", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
danceability = col_features1.slider("Dans edilebilirlik", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
energy = col_features1.slider("Enerji", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
instrumentalness = col_features1.slider("Enstrümantal", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
liveness = col_features1.slider("Canlılık", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
loudness = col_features2.slider("Ses Yüksekliği", min_value=-60, max_value=0, value=-30, step=1)
speechiness = col_features2.slider("Konuşma", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
tempo = col_features2.slider("Tempo", min_value=0, max_value=200, value=100, step=1)
valence = col_features2.slider("Duygu", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

features = np.array([acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence]).reshape(1, -1)


if col_features2.button("Öneri Getir!"):

    distances, indices = pipeline.named_steps['knn'].kneighbors(pipeline.named_steps['scaler'].transform(features), n_neighbors=2)

    recommended_index = indices[0][1]
    recommended_song = df.iloc[recommended_index]

    col_recommendation.image(recommended_song['url'])
    col_recommendation.write(f"**{recommended_song['name']}**")
    col_recommendation.write(f"*{recommended_song['artist']}*")

    if str(recommended_song['preview_url']) != 'nan':
        col_recommendation.audio(recommended_song['preview_url'], format='audio/mp3')

