import yt_dlp
import os
import csv
from openai import OpenAI
from textblob import TextBlob
from transformers import pipeline
import spacy

#need to run python -m spacy download en_core_web_sm !

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
        'format': 'bestaudio',
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
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k = 1,truncation=True)
    
    words = transcript.split()
    third = len(words) // 3
    begin = ' '.join(words[:third])
    middle = ' '.join(words[third:2*third])
    end = ' '.join(words[2*third:])
    
    #max words to 400 so doens't hit token limit 
    begin = ' '.join(begin.split()[:400])
    middle = ' '.join(middle.split()[:400])
    end = ' '.join(end.split()[:400])
    
    begin_emotion = classifier(begin)[0][0]['label']
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

def save_to_csv(results, output_csv="text_results.csv", mode='w'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(script_dir, output_csv)

    fieldnames = ['url', 'query','transcript', 'polarity', 'subjectivity','begin_emotion', 'middle_emotion', 'end_emotion',"people_mentioned","orgs_mentioned","locations_mentioned","events_mentioned","products_mentioned"]

    file_exists = os.path.exists(full_output_path)

    with open(full_output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w' or (mode == 'a' and not file_exists):
            writer.writeheader()
        writer.writerows(results)


#full process
def collect_all(input_csv="../Links/shorts_data/shorts_links_wide.csv",test_first=False):

    if not check_api_key():
        return

    videos = load_csv(input_csv)

    if test_first: #only for testing 
        videos = videos[:1]

    count = 0
    total = len(videos)
    first_save = True

    for video in videos:

        url = video['url']
        query = video['query']

        data = {
            'url': url,
            'query': query,
            'transcript': '',
            'polarity': 0,
            'subjectivity': 0,
            'begin_emotion': '',
            'middle_emotion': '',
            'end_emotion': '',
            'people_mentioned': [],
            'orgs_mentioned': [],
            'locations_mentioned': [],
            'events_mentioned': [],
            'products_mentioned': []
        }

        audio_file, download_success = download_audio(url)
        if not download_success:
            print(f" Download failed - saving empty row")
        else:
            transcript, transcribe_success = get_transcript(audio_file)
            if not transcribe_success or not transcript:
                print(f"Transcription failed - saving empty row")
            elif len(transcript.strip()) < 10:
                word_count = len(transcript.strip().split())
                print(f"Transcript too short ({word_count} words) - saving empty row")
            else:
                # Only collect data once it is valid 
                data = collect_data(transcript)
                data['url'] = url
                data['query'] = query

        mode = 'w' if first_save else 'a'
        save_to_csv([data], output_csv="text_results.csv", mode=mode)
        first_save = False

        count += 1
        print(f"Saved! ({count}/{total} completed)")

    return count


def main():
    check_api_key()
    results = collect_all()
    print(f"Processing complete. Total videos processed: {results}")

if __name__ == "__main__":
    main()
