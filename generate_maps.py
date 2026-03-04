#!/usr/bin/env python3
"""
Do By Friday — Podcast Topic Map Generator

Fetches the RSS feed, downloads episode audio, transcribes with Whisper,
extracts topics/connections with Claude, and generates interactive D3 mind maps.

Usage:
    # Process the latest episode
    python generate_maps.py

    # Process a specific episode number
    python generate_maps.py --episode 442

    # Process multiple episodes
    python generate_maps.py --episode 442 443 444

    # Process all episodes (careful — this is a lot of audio)
    python generate_maps.py --all

    # Skip transcription if you already have a transcript file
    python generate_maps.py --episode 442 --transcript path/to/transcript.txt

    # Use a specific Whisper model (default: medium)
    python generate_maps.py --whisper-model large

Requirements:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    ffmpeg must be installed (for Whisper): brew install ffmpeg
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import anthropic
import feedparser
import requests
import whisper

RSS_URL = "https://feeds.simplecast.com/5nKJV82u"
SCRIPT_DIR = Path(__file__).parent.resolve()
EPISODES_DIR = SCRIPT_DIR / "episodes"
AUDIO_DIR = SCRIPT_DIR / ".audio_cache"
TRANSCRIPT_DIR = SCRIPT_DIR / ".transcripts"
TEMPLATE_PATH = SCRIPT_DIR / "episode_template.html"
EPISODES_JSON = SCRIPT_DIR / "episodes.json"

CATEGORIES = [
    "film", "music", "person", "place", "topic",
    "animal", "tech", "theater", "podcast",
]

EXTRACTION_PROMPT = textwrap.dedent("""\
You are analyzing a transcript of the podcast "Do By Friday" hosted by Alex Cox and Merlin Mann.

Your job is to extract every notable topic, person, film, show, song, place, animal, technology,
or cultural reference discussed — and map the connections between them.

Return ONLY valid JSON with this exact structure (no markdown fences, no commentary):

{
  "nodes": [
    {
      "id": "short_snake_id",
      "label": "Display Name",
      "cat": "film",
      "size": 12,
      "url": "https://en.wikipedia.org/wiki/..."
    }
  ],
  "links": [
    {
      "source": "node_id_1",
      "target": "node_id_2",
      "label": "relationship"
    }
  ]
}

Rules for nodes:
- "id": short unique snake_case identifier (e.g. "kill_bill", "tarantino")
- "label": display name, use \\n for line breaks on long names (max ~16 chars per line)
- "cat": one of: film, music, person, place, topic, animal, tech, theater, podcast
- "size": importance score 7-18 (18 = central topic, 7 = brief mention)
- "url": Wikipedia URL if one exists, omit the field if not

Rules for links:
- "source" and "target" must reference valid node ids
- "label": short description of the relationship (e.g. "directed by", "stars", "same era")
  — omit if the connection is self-evident
- Create links between topics that the hosts actually connected in conversation
- Also create links for implicit connections (e.g. same director, same era, same network)

Aim for 40-100 nodes and rich interconnections. Every node should have at least one link.
Capture the weird tangents and running bits — that's the spirit of the show.
""")


def fetch_feed():
    """Fetch and parse the RSS feed."""
    print("Fetching RSS feed...")
    feed = feedparser.parse(RSS_URL)
    if feed.bozo and not feed.entries:
        print(f"Error parsing feed: {feed.bozo_exception}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(feed.entries)} episodes in feed.")
    return feed


def find_episode(feed, episode_number):
    """Find an episode by number in the feed entries."""
    for entry in feed.entries:
        title = entry.get("title", "")
        # Match patterns like "#442", "Ep 442", "Episode 442", "442:"
        match = re.search(r'#?(?:Ep(?:isode)?\s*)?(\d+)', title)
        if match and int(match.group(1)) == episode_number:
            return entry
    return None


def get_episode_number(entry):
    """Extract episode number from a feed entry."""
    title = entry.get("title", "")
    match = re.search(r'#?(?:Ep(?:isode)?\s*)?(\d+)', title)
    if match:
        return int(match.group(1))
    # Fallback: use itunes_episode if available
    ep = entry.get("itunes_episode")
    if ep:
        return int(ep)
    return None


def get_audio_url(entry):
    """Extract the audio URL from a feed entry."""
    for link in entry.get("links", []):
        if link.get("type", "").startswith("audio/"):
            return link["href"]
    for enc in entry.get("enclosures", []):
        if enc.get("type", "").startswith("audio/"):
            return enc["href"]
    return None


def download_audio(url, episode_number):
    """Download episode audio, with caching."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ext = "mp3"
    filepath = AUDIO_DIR / f"episode_{episode_number}.{ext}"

    if filepath.exists():
        print(f"  Audio already cached: {filepath}")
        return filepath

    print(f"  Downloading audio...")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Downloaded {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    return filepath


def transcribe_audio(audio_path, model_name="medium"):
    """Transcribe audio with OpenAI Whisper."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"{audio_path.stem}.txt"

    if transcript_path.exists():
        print(f"  Transcript already cached: {transcript_path}")
        return transcript_path.read_text()

    print(f"  Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"  Transcribing (this may take a while)...")
    result = model.transcribe(str(audio_path), verbose=False)
    text = result["text"]

    transcript_path.write_text(text)
    print(f"  Transcript saved: {transcript_path}")
    return text


def extract_topics(transcript, episode_number):
    """Use Claude to extract topics and connections from transcript."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        print("Get one at https://console.anthropic.com/", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Truncate if extremely long (Claude has generous limits but let's be safe)
    max_chars = 200_000
    if len(transcript) > max_chars:
        print(f"  Transcript is {len(transcript)} chars, truncating to {max_chars}...")
        transcript = transcript[:max_chars]

    print(f"  Sending transcript to Claude for topic extraction...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Here is the transcript of Do By Friday episode #{episode_number}:\n\n"
                    f"{transcript}\n\n"
                    f"{EXTRACTION_PROMPT}"
                ),
            }
        ],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if Claude added them anyway
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Save the raw response for debugging
        debug_path = SCRIPT_DIR / f".debug_response_{episode_number}.txt"
        debug_path.write_text(raw)
        print(f"  Error parsing Claude's JSON response: {e}", file=sys.stderr)
        print(f"  Raw response saved to {debug_path}", file=sys.stderr)
        sys.exit(1)

    nodes = data.get("nodes", [])
    links = data.get("links", [])
    print(f"  Extracted {len(nodes)} topics and {len(links)} connections.")
    return nodes, links


def validate_graph(nodes, links):
    """Validate and clean up the graph data."""
    node_ids = {n["id"] for n in nodes}

    # Remove links that reference missing nodes
    valid_links = []
    for link in links:
        if link["source"] in node_ids and link["target"] in node_ids:
            valid_links.append(link)
        else:
            missing = []
            if link["source"] not in node_ids:
                missing.append(link["source"])
            if link["target"] not in node_ids:
                missing.append(link["target"])
            print(f"  Warning: dropping link with missing node(s): {missing}")

    # Validate categories
    for node in nodes:
        if node.get("cat") not in CATEGORIES:
            print(f"  Warning: unknown category '{node.get('cat')}' for node '{node['id']}', defaulting to 'topic'")
            node["cat"] = "topic"

        # Ensure size is in range
        node["size"] = max(7, min(18, node.get("size", 10)))

    return nodes, valid_links


def generate_html(nodes, links, episode_number):
    """Generate the self-contained HTML mind map file."""
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE_PATH.read_text()

    # Format nodes and links as JavaScript-friendly JSON
    nodes_json = json.dumps(nodes, indent=2, ensure_ascii=False)
    links_json = json.dumps(links, indent=2, ensure_ascii=False)

    html = template.replace("{{EPISODE_NUMBER}}", str(episode_number))
    html = html.replace("{{NODES_JSON}}", nodes_json)
    html = html.replace("{{LINKS_JSON}}", links_json)

    out_path = EPISODES_DIR / f"{episode_number}.html"
    out_path.write_text(html)
    print(f"  Mind map saved: {out_path}")
    return out_path


def update_episodes_json(episode_number, title, date, num_nodes):
    """Add or update the episode in episodes.json."""
    data = json.loads(EPISODES_JSON.read_text())

    # Check if episode already exists
    for ep in data["episodes"]:
        if ep["number"] == episode_number:
            ep["title"] = title
            ep["date"] = date
            ep["description"] = f"{num_nodes} topics mapped."
            break
    else:
        data["episodes"].append({
            "number": episode_number,
            "title": title,
            "date": date,
            "description": f"{num_nodes} topics mapped.",
        })

    # Sort by episode number descending
    data["episodes"].sort(key=lambda e: e["number"], reverse=True)

    EPISODES_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"  Updated episodes.json")


def process_episode(feed, episode_number, whisper_model="medium", transcript_path=None):
    """Full pipeline for one episode."""
    print(f"\n{'='*60}")
    print(f"Processing Episode #{episode_number}")
    print(f"{'='*60}")

    entry = find_episode(feed, episode_number)
    if not entry:
        print(f"  Episode #{episode_number} not found in RSS feed.", file=sys.stderr)
        return False

    title = entry.get("title", f"Episode {episode_number}")
    date = ""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        t = entry.published_parsed
        date = f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"

    print(f"  Title: {title}")
    print(f"  Date:  {date}")

    # Step 1: Get transcript
    if transcript_path:
        print(f"  Using provided transcript: {transcript_path}")
        transcript = Path(transcript_path).read_text()
    else:
        audio_url = get_audio_url(entry)
        if not audio_url:
            print(f"  Error: No audio URL found for this episode.", file=sys.stderr)
            return False
        print(f"  Audio: {audio_url[:80]}...")

        audio_file = download_audio(audio_url, episode_number)
        transcript = transcribe_audio(audio_file, whisper_model)

    # Step 2: Extract topics with Claude
    nodes, links = extract_topics(transcript, episode_number)
    nodes, links = validate_graph(nodes, links)

    # Step 3: Generate HTML
    generate_html(nodes, links, episode_number)

    # Step 4: Update episodes.json
    update_episodes_json(episode_number, title, date, len(nodes))

    print(f"\n  Done! Open episodes/{episode_number}.html in a browser to view.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate Do By Friday topic connection maps from podcast episodes."
    )
    parser.add_argument(
        "--episode", "-e",
        nargs="+",
        type=int,
        help="Episode number(s) to process (default: latest)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all episodes in the feed",
    )
    parser.add_argument(
        "--transcript", "-t",
        type=str,
        help="Path to an existing transcript file (skips download + transcription)",
    )
    parser.add_argument(
        "--whisper-model", "-w",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )

    args = parser.parse_args()

    # Preflight checks
    if not TEMPLATE_PATH.exists():
        print(f"Error: Template not found at {TEMPLATE_PATH}", file=sys.stderr)
        sys.exit(1)
    if not EPISODES_JSON.exists():
        print(f"Error: episodes.json not found at {EPISODES_JSON}", file=sys.stderr)
        sys.exit(1)

    feed = fetch_feed()

    if args.all:
        episodes = []
        for entry in feed.entries:
            num = get_episode_number(entry)
            if num is not None:
                episodes.append(num)
        episodes.sort()
        print(f"Will process {len(episodes)} episodes: {episodes[0]}–{episodes[-1]}")
    elif args.episode:
        episodes = args.episode
    else:
        # Default: latest episode
        latest = feed.entries[0]
        num = get_episode_number(latest)
        if num is None:
            print("Could not determine episode number from latest entry.", file=sys.stderr)
            print(f"Title: {latest.get('title')}", file=sys.stderr)
            sys.exit(1)
        episodes = [num]

    success = 0
    for ep_num in episodes:
        transcript = args.transcript if len(episodes) == 1 else None
        if process_episode(feed, ep_num, args.whisper_model, transcript):
            success += 1

    print(f"\n{'='*60}")
    print(f"Finished: {success}/{len(episodes)} episodes processed successfully.")
    if success:
        print(f"Preview: python3 -m http.server 8000  →  http://localhost:8000")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
