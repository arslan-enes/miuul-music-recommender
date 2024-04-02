import requests
import os
import pandas as pd
import json
import time
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

PLAYLISTS = [
    "37i9dQZF1DXcBWIGoYBM5M",
    "37i9dQZEVXbMDoHDwVN2tF", 
    "37i9dQZF1DX0XUsuxWHRQd",
    "37i9dQZF1DX10zKzsJ2jva",
    "37i9dQZF1DWXRqgorJj26U",
    "37i9dQZF1DX4o1oenSJRJd",
    "37i9dQZF1DWWMOmoXKqHTD",
    "37i9dQZF1DX4UtSsGT1Sbe",
    "37i9dQZF1DWY7IeIP1cdjF",
    "37i9dQZF1DX76Wlfdnj7AP",
    "37i9dQZF1DWWOaP4H0w5b0",
    "37i9dQZF1DXbITWG1ZJKYt"
]

def get_songs_from_playlist(playlist_id):

    """
    This function takes a Spotify playlist id and returns a DataFrame with the following columns:
    - url: url of the album cover
    - artist: name of the artist
    - id: id of the song
    - name: name of the song
    - preview_url: url of the preview
    - release_date: release date of the song
    - album_id: id of the album
    """

    json_data = json.loads(requests.get(f'https://api.spotify.com/v1/playlists/{playlist_id}?fields=tracks.items.track(id, name, preview_url, popularity, duration_ms, artists(name,id), album(images.url, release_date, id))', 
                                        headers={'Authorization': f'Bearer {ACCESS_TOKEN}'}).text)


    df = pd.DataFrame(columns=['url', 'artist', 'id', 'name', 'preview_url', 'release_date', 'album_id', 'popularity', 'duration_ms', 'artist_id'])


    for index, item in enumerate(json_data['tracks']['items']):
        df.loc[index] = [item['track']['album']['images'][0]['url'], 
                        item['track']['artists'][0]['name'], 
                        item['track']['id'], 
                        item['track']['name'], 
                        item['track']['preview_url'], 
                        item['track']['album']['release_date'],
                        item['track']['album']['id'],
                        item['track']['popularity'],
                        item['track']['duration_ms'],
                        item['track']['artists'][0]['id']]

    return df


def get_songs_from_album(album_id):

    """
    This function takes a Spotify album id and returns a DataFrame with the following columns:
    - url: url of the album cover
    - artist: name of the artist
    - id: id of the song
    - name: name of the song
    - preview_url: url of the preview
    - release_date: release date of the song
    - album_id: id of the album
    - popularity: popularity of the song
    - duration_ms: duration of the song
    - artist_id: id of the artist
    """

    response = requests.get(f'https://api.spotify.com/v1/albums/{album_id}/tracks?fields=items.id', headers={'Authorization': f'Bearer {ACCESS_TOKEN}'})

    json_data = json.loads(response.text)
    status_code = response.status_code

    print(f'Status code: {status_code}')

    df = pd.DataFrame(columns=['url', 'artist', 'id', 'name', 'preview_url', 'release_date', 'album_id', 'popularity', 'duration_ms', 'artist_id'])

    for item in json_data['items']:
        
        song = json.loads(requests.get(f'https://api.spotify.com/v1/tracks/{item["id"]}', headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}).text)
        
        df.loc[len(df)] = [song['album']['images'][0]['url'],
                            song['artists'][0]['name'],
                            song['id'],
                            song['name'],
                            song['preview_url'],
                            song['album']['release_date'],
                            song['album']['id'],
                            song['popularity'],
                            song['duration_ms'],
                            song['artists'][0]['id']]
        

    return df


def get_all_songs():
    
    """
    This function returns a DataFrame with the following columns:
    - url: url of the album cover
    - artist: name of the artist
    - id: id of the song
    - name: name of the song
    - preview_url: url of the preview
    - release_date: release date of the song
    - album_id: id of the album
    - popularity: popularity of the song
    - duration_ms: duration of the song
    - artist_id: id of the artist
    """
    df = pd.DataFrame(columns=['url', 'artist', 'id', 'name', 'preview_url', 'release_date', 'album_id', 'popularity', 'duration_ms', 'artist_id'])

    for playlist in PLAYLISTS:
        df = pd.concat([df, get_songs_from_playlist(playlist)])

        print(f'Playlist {playlist} done. {len(df)} songs collected.')
    
    for album in df['album_id'].unique():
        try:
            df = pd.concat([df, get_songs_from_album(album)])
            print(f'Album {album} done. {len(df)} songs collected.')
        except:
            print(f'Album {album} failed.')
            continue

    df.drop_duplicates(inplace=True)

    return df

def get_song_features(song_id):

    """
    This function takes a Spotify song id and returns a list with the following features:
    - id: id of the song
    - acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
    - danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
    - energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.
    - instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.
    - liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.
    - loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks.
    - speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value.
    - tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
    - valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
    """

    response = requests.get(f'https://api.spotify.com/v1/audio-features/{song_id}', headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'})

    json_data = json.loads(response.text)
    status_code = response.status_code

    print(f'Status code: {status_code}')

    time.sleep(0.5)

    return [json_data['id'],
            json_data['acousticness'],
            json_data['danceability'],
            json_data['energy'],
            json_data['instrumentalness'],
            json_data['liveness'],
            json_data['loudness'],
            json_data['speechiness'],
            json_data['tempo'],
            json_data['valence']]


if __name__ == '__main__':
    df = pd.read_csv("all_songs.csv")

    df_features = pd.read_csv("all_songs_features.csv")

    pd.merge(df, df_features, on='id', how="left").to_csv("data.csv", index=False)