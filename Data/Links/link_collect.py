"""
YouTube Shorts Collector (English Only)
-----------------------------------------
Collects YouTube Shorts links in English using the YouTube Data API v3
and a DuckDuckGo fallback scraper. Results are filtered to English content
via API parameters and heuristic keyword filtering.

Outputs CSV: query, category_type, url
"""

import os
import time
import random
import requests
import csv
from urllib.parse import quote

# Import credentials
try:
    from credentials import YOUTUBE_API_KEY
except ImportError:
    YOUTUBE_API_KEY = ""

# === CONFIGURATION ===
OUTPUT_DIR = "shorts_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "shorts_links_english.csv")
RESULTS_PER_QUERY = 20
DELAY_BETWEEN_QUERIES = (2, 4)

SEARCH_QUERIES = {
    "core": [
        "funny Shorts",
        "educational Shorts",
        "recipe Shorts",
        "music Shorts",
        "fashion Shorts",
        "sports Shorts",
        "motivational Shorts",
        "gaming Shorts",
        "travel Shorts",
        "tech Shorts"
    ],
    "trending": [
        "viral Shorts",
        "trending Shorts",
        "ai Shorts",
        "reaction Shorts",
        "storytime Shorts",
        "remix Shorts",
        "meme Shorts",
        "challenge Shorts",
        "random Shorts"
    ]
}

FALLBACK_SITE_SUFFIX = "site:youtube.com/shorts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_api_key():
    if not YOUTUBE_API_KEY:
        print("No API key found. Using fallback mode.")
        return False

    test_url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&q=test&type=video&maxResults=1&key={YOUTUBE_API_KEY}"
    )
    resp = requests.get(test_url)
    if resp.status_code == 200:
        print("YouTube API key loaded successfully.")
        return True
    else:
        print(f"YouTube API key test failed ({resp.status_code})")
        return False


def search_youtube_api(query, max_results):
    """Use YouTube API to get English-language Shorts."""
    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY,
        "videoDuration": "short",
        "relevanceLanguage": "en",  # bias toward English
        "videoCaption": "none"      # avoid non-English subtitles
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"YouTube API error for '{query}': {response.status_code}")
        return []

    data = response.json()
    results = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        title = snippet.get("title", "").lower()
        description = snippet.get("description", "").lower()
        # simple English heuristic (if text contains mostly ASCII)
        english_ratio = sum(c.isascii() for c in title + description) / max(1, len(title + description))
        if english_ratio < 0.9:
            continue  # skip non-English videos

        video_id = item["id"].get("videoId")
        if video_id:
            results.append(f"https://www.youtube.com/shorts/{video_id}")
    return results


def fallback_scrape(query, max_results):
    """Fallback method using DuckDuckGo search, filters likely English content."""
    query_with_site = f"{query} {FALLBACK_SITE_SUFFIX}"
    base_url = f"https://duckduckgo.com/html/?q={quote(query_with_site)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(base_url, headers=headers)
    if response.status_code != 200:
        return []

    import re
    urls = re.findall(r"https://www\.youtube\.com/shorts/[a-zA-Z0-9_-]{6,}", response.text)
    urls = list(dict.fromkeys(urls))  # deduplicate

    # crude English filtering heuristic (keep first N)
    english_like = [u for u in urls if any(word in u.lower() for word in ["funny", "meme", "reaction", "story", "challenge", "shorts"])]
    return english_like[:max_results]


def collect_all_shorts(queries_dict, use_api):
    all_links = []

    for category_type, queries in queries_dict.items():
        print(f"\nCollecting for category type: {category_type.upper()}")
        for query in queries:
            if use_api:
                results = search_youtube_api(query, RESULTS_PER_QUERY)
            else:
                results = fallback_scrape(query, RESULTS_PER_QUERY)

            for url in results:
                all_links.append({
                    "query": query,
                    "category_type": category_type,
                    "url": url
                })

            print(f"  Found {len(results)} English Shorts for '{query}'")
            time.sleep(random.uniform(*DELAY_BETWEEN_QUERIES))

    # Deduplicate by URL
    unique_links = {entry["url"]: entry for entry in all_links}.values()

    # Save CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "category_type", "url"])
        writer.writeheader()
        writer.writerows(unique_links)

    print(f"\nSaved {len(unique_links)} total unique English Shorts to {OUTPUT_FILE}")
    print("Dataset ready for multimodal feature extraction.")


def main():
    print("Starting YouTube Shorts (English-only) collection...")
    use_api = check_api_key()
    collect_all_shorts(SEARCH_QUERIES, use_api)
    print("\nCollection complete.")


if __name__ == "__main__":
    main()
