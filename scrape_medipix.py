# scrape_medpix.py
import os
import requests
from bs4 import BeautifulSoup
import json
import urllib.request
from tqdm import tqdm

# CONFIGURATION
OUTPUT_FOLDER = "data/images/"
META_FILE = "data/image_meta.json"
BASE_URL = "https://medpix.nlm.nih.gov"
SEARCH_URL = "https://medpix.nlm.nih.gov/search?filters=all"
MAX_IMAGES = 1000
MAX_TOTAL_SIZE_MB = 2500  # ~2.5 GB

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
image_metadata = []
total_size_mb = 0

headers = {"User-Agent": "Mozilla/5.0"}
print("\n🔍 Starting MedPix Scraper...")

# Get the first search page
response = requests.get(SEARCH_URL, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

case_links = list(set(BASE_URL + a['href'] for a in soup.select("a") if "/case" in a.get('href', '')))

img_counter = 0

for link in tqdm(case_links[:2000], desc="🔗 Scanning MedPix Cases"):
    try:
        case_resp = requests.get(link, headers=headers, timeout=10)
        case_soup = BeautifulSoup(case_resp.content, 'html.parser')

        # Extract title/description
        title_tag = case_soup.find("h1")
        caption = title_tag.get_text(strip=True) if title_tag else "No Caption"

        # Diagnosis or extra text
        diag_tag = case_soup.find("div", class_="diagnosis")
        diagnosis = diag_tag.get_text(strip=True) if diag_tag else "Unknown Diagnosis"

        img_tags = case_soup.select("img")
        for img_tag in img_tags:
            if 'src' in img_tag.attrs and "/images" in img_tag['src']:
                img_url = BASE_URL + img_tag['src']
                img_ext = img_url.split(".")[-1].split("?")[0]
                filename = f"medpix_{img_counter+1:04d}.{img_ext}"
                save_path = os.path.join(OUTPUT_FOLDER, filename)

                # Download image
                urllib.request.urlretrieve(img_url, save_path)
                size_mb = os.path.getsize(save_path) / (1024 * 1024)

                # Check size constraint
                total_size_mb += size_mb
                if total_size_mb > MAX_TOTAL_SIZE_MB:
                    os.remove(save_path)
                    print(f"⚠️ Stopped scraping: size limit exceeded ~{MAX_TOTAL_SIZE_MB}MB")
                    raise StopIteration

                # Save metadata
                image_metadata.append({
                    "image_file": filename,
                    "caption": caption,
                    "diagnosis": diagnosis,
                    "source_url": link
                })

                img_counter += 1
                if img_counter >= MAX_IMAGES:
                    raise StopIteration

    except StopIteration:
        break
    except Exception as e:
        continue

# Save metadata
with open(META_FILE, "w") as f:
    json.dump(image_metadata, f, indent=4)

print(f"\n✅ Done. Total images downloaded: {img_counter}")
print(f"✅ Total size: ~{total_size_mb:.2f} MB")
print(f"✅ Metadata saved to: {META_FILE}")
