import yt_dlp
import os
import csv
from openai import OpenAI
from textblob import TextBlob
from transformers import pipeline
import spacy

#python -m spacy download en_core_web_sm 

try:
    from credentials import OPENAI_API_KEY 
except ImportError:
    OPENAI_API_KEY = ""

def check_api_key():
    if not OPENAI_API_KEY:
        print("No api key found for whisper")
        return False
    return True

def load_csv(csv_file="shorts_links_wide.csv"):
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"Loaded {len(rows)} videos from {csv_file}")
        return rows
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []


def download_audio(url, output_path="temp_audio_downloads"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(script_dir, output_path)

    os.makedirs(full_output_path, exist_ok=True)

    ydl_dowload_settings = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),  
    }

    try:
        ydl = yt_dlp.YoutubeDL(ydl_dowload_settings)
        print(f"Downloading audio from: {url}")
        info = ydl.extract_info(url, download=True)
        title = info['title']
        ext = info['ext'] 
        audio_file = os.path.join(full_output_path, f"{title}.{ext}")

        print(f"Downloaded audio to: {audio_file}")
        return audio_file, True
    except Exception as e:
        print(f"Error download {url}: {e}")
        return None, False

#whisper transcription
def get_transcript(audio_file):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        file = open(audio_file, "rb")
        transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=file,
                response_format="json"
            )
        file.close()
        
        print(f"Transcription complete: {audio_file}")
        return transcript.text, True
    except Exception as e:
        print(f"Error transcribing {audio_file}: {e}")
        return None, False
    
#collect data from transcript 
#focus on subject and context
def collect_data(transcript):

    #sentiment/polarity
    print("Analyze sentiment")
    blob = TextBlob(transcript)
    sentiment = blob.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    
    #Emotion(Direction)
    print("Analyze emotions")
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k = 1)
    
    sentences = transcript.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    third = len(sentences) // 3
    beginning = ' '.join(sentences[:third])
    middle = ' '.join(sentences[third:2*third])
    end = ' '.join(sentences[2*third:])

    begin_emotion = classifier(beginning)[0][0]['label']
    middle_emotion = classifier(middle)[0][0]['label']
    end_emotion = classifier(end)[0][0]['label']

    #Mentions
    print("Analyze mentions")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(transcript)

    people_mentioned = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    orgs_mentioned = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    locations_mentioned = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    events_mentioned = [ent.text for ent in doc.ents if ent.label_ == 'EVENT']
    products_mentioned = [ent.text for ent in doc.ents if ent.label_ == 'PRODUCT']
    
    return{
        "transcript": transcript,
        "polarity" : polarity,
        "subjectivity": subjectivity,
        "begin_emotion": begin_emotion, 
        "middle_emotion": middle_emotion,  
        "end_emotion": end_emotion,  
        "people_mentioned": people_mentioned,
        "orgs_mentioned": orgs_mentioned,
        "locations_mentioned": locations_mentioned,
        "events_mentioned": events_mentioned,
        "products_mentioned": products_mentioned
    }

#full process
def collect_all(input_csv="../Links/shorts_data/shorts_links_wide.csv",test_first=True):

    if not check_api_key():
        return

    videos = load_csv(input_csv)

    if test_first: #only for testing 
        videos = videos[:1]

    results_before_CSV = []

    for video in videos:

        url = video['url']
        query = video['query']

        result = {'url': url, 'query': query}

        audio_file, download_success = download_audio(url)
        if not download_success:
            print(f"Skipping {url} - download failed")
            continue

        transcript, transcribe_success = get_transcript(audio_file)
        if not transcribe_success:
            print(f"Skipping {url} - transcription failed")
            continue

        data = collect_data(transcript)
        result.update(data)
        results_before_CSV.append(result)
        print(f"Processed video")

    return results_before_CSV


def save_to_csv(results, output_csv="text_results.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(script_dir, output_csv)

    fieldnames = ['url', 'query','transcript', 'polarity', 'subjectivity','begin_emotion', 'middle_emotion', 'end_emotion',"people_mentioned","orgs_mentioned","locations_mentioned","events_mentioned","products_mentioned"]

    with open(full_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Saved to CSV")


def main():
    check_api_key()
    results = collect_all(test_first=True)
    save_to_csv(results)


if __name__ == "__main__":
    main()
