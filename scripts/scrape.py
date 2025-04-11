import requests
from bs4 import BeautifulSoup
import csv
import time

GENIUS_API_TOKEN = "API-KEY"
HEADERS = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}


def get_artist_id(artist_name):
    url = "https://api.genius.com/search"
    params = {"q": artist_name}
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()

    for hit in data["response"]["hits"]:
        if artist_name.lower() in hit["result"]["primary_artist"]["name"].lower():
            return hit["result"]["primary_artist"]["id"]

    return None


def get_artist_songs(artist_id, max_songs=50):
    songs = []
    page = 1

    while len(songs) < max_songs:
        url = f"https://api.genius.com/artists/{artist_id}/songs"
        params = {"per_page": 50, "page": page, "sort": "popularity"}
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()

        page_songs = data["response"]["songs"]
        if not page_songs:
            break

        for song in page_songs:
            songs.append(
                {
                    "title": song["title"],
                    "artist": song["primary_artist"]["name"],
                    "url": song["url"],
                }
            )
            if len(songs) >= max_songs:
                break

        page += 1

    return songs


def scrape_lyrics(url):
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        lyrics_divs = soup.find_all(
            "div", class_=lambda x: x and "Lyrics__Container" in x
        )
        lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])
        return lyrics.strip()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def save_lyrics_for_artists(
    artist_list, output_csv="multi_artist_lyrics.csv", max_songs=50
):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "artist", "url", "lyrics"])
        writer.writeheader()

        for artist_name in artist_list:
            print(f"\n🔍 Processing artist: {artist_name}")
            artist_id = get_artist_id(artist_name)
            if not artist_id:
                print(f"❌ Artist not found: {artist_name}")
                continue

            songs = get_artist_songs(artist_id, max_songs=max_songs)

            for song in songs:
                print(f"🎵 Getting lyrics for: {song['title']} by {song['artist']}")
                lyrics = scrape_lyrics(song["url"])
                writer.writerow(
                    {
                        "title": song["title"],
                        "artist": song["artist"],
                        "url": song["url"],
                        "lyrics": lyrics,
                    }
                )
                time.sleep(1)  # Prevent IP blocking


if __name__ == "__main__":
    artists = ["Kendrick Lamar", "J. Cole", "Drake", "Nicki Minaj"]
    save_lyrics_for_artists(artists, output_csv="rap_lyrics_dataset.csv", max_songs=100)
