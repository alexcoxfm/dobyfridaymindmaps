#!/usr/bin/env python3
"""
Do By Friday — Podcast Topic Map Generator

Reads each episode's transcript from the Do By Friday Archives site
(https://dobyfridayarchives.pages.dev), has Claude pull out the topics and
how they connect, and writes an interactive D3 mind map for the site.

No audio download or transcription is needed: the archive already has a
timestamped, speaker-labelled transcript for every episode.

By default the topic step runs through Claude Code on this Mac (`claude -p`),
using the Claude account you're signed in with, so no API key is needed and
nothing goes to the billed API. API keys in your environment are removed for
those calls. When Claude Code reports a usage limit, the script stops; run the
same command again later and it skips episodes that already have a map.

Usage:
    # Free check: fetch and parse transcripts, no Claude calls, no API key needed
    python generate_maps.py --episode 1 250 472 --dry-run

    # Make maps for specific episodes (ranges work too)
    python generate_maps.py --episode 442 443
    python generate_maps.py --episode 400-410

    # Latest episode on the archive
    python generate_maps.py

    # Every numbered episode on the archive (skips episodes that already have a map)
    python generate_maps.py --all

    # Use the billed Anthropic API instead of Claude Code (needs ANTHROPIC_API_KEY;
    # --all then only prints a cost estimate unless you add --yes)
    python generate_maps.py --episode 442 --engine api

    # Use your own transcript file for one episode
    python generate_maps.py --episode 442 --transcript path/to/transcript.txt

    # Old pipeline: download audio from the RSS feed and transcribe with Whisper
    python generate_maps.py --episode 442 --source rss

Requirements:
    pip install -r requirements.txt
    Claude Code installed and signed in (run `claude` once)
    Only for --engine api: pip install anthropic, export ANTHROPIC_API_KEY=...
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import requests

ARCHIVE_URL = "https://dobyfridayarchives.pages.dev"
RSS_URL = "https://feeds.simplecast.com/5nKJV82u"
SCRIPT_DIR = Path(__file__).parent.resolve()
EPISODES_DIR = SCRIPT_DIR / "episodes"
AUDIO_DIR = SCRIPT_DIR / ".audio_cache"
TRANSCRIPT_DIR = SCRIPT_DIR / ".transcripts"
TEMPLATE_PATH = SCRIPT_DIR / "episode_template.html"
EPISODES_JSON = SCRIPT_DIR / "episodes.json"

DEFAULT_MODEL = "claude-opus-5-5"
# Standard API prices in US dollars per million tokens (input, output),
# used only for the cost printout.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

HTTP_HEADERS = {"User-Agent": "dbf-mindmaps/1.0 (+https://alexcoxfm.github.io/dobyfridaymindmaps/)"}

CATEGORIES = [
    "film", "music", "person", "place", "topic",
    "animal", "tech", "theater", "podcast",
]

# Structured output: the API guarantees Claude's reply matches this shape.
GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "cat": {"type": "string", "enum": CATEGORIES},
                    "size": {"type": "integer"},
                    "url": {"type": "string"},
                },
                "required": ["id", "label", "cat", "size", "url"],
                "additionalProperties": False,
            },
        },
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["source", "target", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes", "links"],
    "additionalProperties": False,
}

EXTRACTION_PROMPT = textwrap.dedent("""\
You are analyzing a transcript of the podcast "Do By Friday" hosted by Alex Cox and Merlin Mann.
The transcript is machine-generated, so names and titles are sometimes misheard, and the
speaker labels are assigned by software and can be wrong or generic ("Speaker A").
Use the correct spelling of names and titles in your labels.

Extract every notable topic, person, film, show, song, place, animal, technology,
or cultural reference discussed, and map the connections between them.

Rules for nodes:
- "id": short unique snake_case identifier (e.g. "kill_bill", "tarantino")
- "label": display name, use \\n for line breaks on long names (max ~16 chars per line)
- "cat": one of: film, music, person, place, topic, animal, tech, theater, podcast
  (TV shows go under "film")
- "size": importance score 7-18 (18 = central topic, 7 = brief mention)
- "url": Wikipedia URL if you are confident one exists, otherwise an empty string

Rules for links:
- "source" and "target" must reference valid node ids
- "label": short description of the relationship (e.g. "directed by", "stars", "same era"),
  or an empty string if the connection is self-evident
- Link topics that the hosts actually connected in conversation
- Also link implicit connections (e.g. same director, same era, same network)

Aim for 40-100 nodes and rich interconnections. Every node should have at least one link.
Capture the weird tangents and running bits — that's the spirit of the show.
""")


class GenerationError(Exception):
    """One episode failed; the run carries on with the next one."""


# ---------------------------------------------------------------------------
# Archive transcripts
# ---------------------------------------------------------------------------

def http_get(url, retries=3):
    """GET a page, retrying a couple of times on network errors."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise GenerationError(f"Could not fetch {url}: {e}")
            time.sleep(2 * (attempt + 1))


def episode_page_url(episode_number):
    return f"{ARCHIVE_URL}/e{episode_number:03d}.html"


def archive_episode_numbers():
    """All numbered episodes linked from the archive's home page."""
    html = http_get(f"{ARCHIVE_URL}/")
    if not html:
        raise GenerationError("Could not load the archive home page.")
    nums = {int(m) for m in re.findall(r'href="(?:\./|/)?e(\d{3,})(?:\.html)?"', html)}
    if not nums:
        raise GenerationError("Found no episode links on the archive home page.")
    return sorted(nums)


TIMESTAMP_RE = re.compile(r"\b\d{1,2}(?::\d{2}){1,2}\b")
EPISODE_LINE_RE = re.compile(
    r"Episode\s+(\d+)\s*·\s*([A-Z][a-z]+ \d{1,2}, \d{4})\s*·\s*(\d{1,2}:\d{2}(?::\d{2})?)"
)


class ArchivePageParser(HTMLParser):
    """
    Pulls the transcript out of an archive episode page.

    The archive marks each transcript paragraph with an element whose id is
    "t-<seconds>" (the timestamp and speaker live in it), and wraps each
    sentence in a <span data-s="...">. Text in those spans is the transcript;
    the rest of the anchor element is the heading (timestamp, speaker name,
    "Copy link" button).
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs = []      # dicts: start, heading, spoken, loose
        self.h1 = ""
        self.title = ""
        self.all_text = []
        self._in_h1 = False
        self._in_title = False
        self._skip_depth = 0      # inside <script>/<style>
        self._anchor_tag = None
        self._anchor_depth = 0
        self._span_depth = 0      # depth inside a span[data-s]

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self._skip_depth += 1
            return
        if tag == "h1":
            self._in_h1 = True
        if tag == "title":
            self._in_title = True

        m = re.fullmatch(r"t-(\d+)", attrs.get("id") or "")
        if m and self._anchor_tag is None:
            self.paragraphs.append(
                {"start": int(m.group(1)), "heading": [], "spoken": [], "loose": []}
            )
            self._anchor_tag = tag
            self._anchor_depth = 1
            return
        if self._anchor_tag == tag:
            self._anchor_depth += 1

        if tag == "span":
            if self._span_depth:
                self._span_depth += 1
            elif "data-s" in attrs:
                self._span_depth = 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "h1":
            self._in_h1 = False
        if tag == "title":
            self._in_title = False
        if tag == "span" and self._span_depth:
            self._span_depth -= 1
        if self._anchor_tag == tag:
            self._anchor_depth -= 1
            if self._anchor_depth == 0:
                self._anchor_tag = None

    def handle_data(self, data):
        if self._skip_depth:
            return
        self.all_text.append(data)
        if self._in_h1:
            self.h1 += data
        if self._in_title:
            self.title += data
        if not self.paragraphs:
            return
        para = self.paragraphs[-1]
        if self._span_depth:
            para["spoken"].append(data)
        elif self._anchor_tag is not None:
            para["heading"].append(data)
        else:
            para["loose"].append(data)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_heading_noise(text):
    text = re.sub(r"Copy link.*", "", text, flags=re.I)
    return TIMESTAMP_RE.sub("", text)


def speaker_from_heading(heading):
    text = clean(strip_heading_noise(heading).strip(" ·:-–—|"))
    if text and len(text) <= 40 and len(text.split()) <= 4:
        return text
    return None


def parse_archive_page(html):
    """Return episode metadata and transcript paragraphs from an archive page."""
    parser = ArchivePageParser()
    parser.feed(html)
    parser.close()

    use_spans = any(clean(" ".join(p["spoken"])) for p in parser.paragraphs)
    paragraphs = []
    speaker = None
    for p in parser.paragraphs:
        heading = clean(" ".join(p["heading"]))
        named = speaker_from_heading(heading)
        if use_spans:
            body = clean(" ".join(p["spoken"]))
        else:
            # No sentence spans: the paragraph text is whatever follows the anchor.
            body = clean(" ".join(p["loose"]))
            if not body:
                body, named = clean(strip_heading_noise(heading)), None
        if named:
            speaker = named  # paragraphs with no name continue the previous speaker
        if body:
            paragraphs.append({"start": p["start"], "speaker": speaker, "text": body})

    full_text = clean(" ".join(parser.all_text))
    meta = {"heading": clean(parser.h1) or clean(parser.title.split("·")[0])}
    m = EPISODE_LINE_RE.search(full_text)
    if m:
        meta["number"] = int(m.group(1))
        try:
            meta["date"] = datetime.strptime(m.group(2), "%B %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            meta["date"] = ""
        meta["duration"] = m.group(3)
    title = re.sub(r"^\d+\.\s*", "", meta["heading"])
    meta["title"] = title or meta["heading"]
    return meta, paragraphs


def transcript_text(paragraphs):
    """Plain transcript for Claude: one line per speaker turn, no timestamps."""
    lines = []
    for p in paragraphs:
        who = p["speaker"] or "Unknown"
        if lines and lines[-1][0] == who:
            lines[-1][1].append(p["text"])
        else:
            lines.append((who, [p["text"]]))
    return "\n\n".join(f"{who}: {' '.join(parts)}" for who, parts in lines)


def hms_to_seconds(value):
    total = 0
    for part in value.split(":"):
        total = total * 60 + int(part)
    return total


def fmt_seconds(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fetch_archive_episode(episode_number, refresh=False):
    """Fetch an episode's transcript from the archive (cached in .transcripts/)."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    cache = TRANSCRIPT_DIR / f"archive_e{episode_number:03d}.json"
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
        return data["meta"], data["paragraphs"]

    url = episode_page_url(episode_number)
    html = http_get(url)
    if html is None:
        raise GenerationError(f"No archive page for episode {episode_number} ({url}).")
    meta, paragraphs = parse_archive_page(html)
    if not paragraphs:
        raise GenerationError(f"Found no transcript on {url}. The page layout may have changed.")
    if meta.get("number") not in (None, episode_number):
        raise GenerationError(f"{url} says it is episode {meta['number']}, not {episode_number}.")
    meta["source_url"] = url
    cache.write_text(json.dumps({"meta": meta, "paragraphs": paragraphs}, ensure_ascii=False))
    return meta, paragraphs


def describe_transcript(meta, paragraphs, text):
    """Print a short health check of a fetched transcript."""
    words = len(text.split())
    speakers = {}
    for p in paragraphs:
        who = p["speaker"] or "Unknown"
        speakers[who] = speakers.get(who, 0) + 1
    top = ", ".join(f"{k} {v}" for k, v in sorted(speakers.items(), key=lambda kv: -kv[1])[:5])
    last = paragraphs[-1]["start"]
    coverage = ""
    if meta.get("duration"):
        dur = hms_to_seconds(meta["duration"])
        if dur:
            coverage = f" of {meta['duration']} ({last / dur:.0%} covered)"
    print(f"  Transcript: {len(paragraphs)} paragraphs, {words:,} words (~{int(words * 1.35):,} tokens)")
    print(f"  Last line at {fmt_seconds(last)}{coverage}")
    print(f"  Paragraphs by speaker: {top}")
    for line in text.split("\n\n")[:2]:
        print(f"    | {line[:110]}")


# ---------------------------------------------------------------------------
# RSS + Whisper (old pipeline, used with --source rss)
# ---------------------------------------------------------------------------

def fetch_feed():
    import feedparser  # only needed for --source rss

    print("Fetching RSS feed...")
    feed = feedparser.parse(RSS_URL)
    if feed.bozo and not feed.entries:
        print(f"Error parsing feed: {feed.bozo_exception}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(feed.entries)} episodes in feed.")
    return feed


def get_episode_number(entry):
    title = entry.get("title", "")
    match = re.search(r'#?(?:Ep(?:isode)?\s*)?(\d+)', title)
    if match:
        return int(match.group(1))
    ep = entry.get("itunes_episode")
    return int(ep) if ep else None


def find_episode(feed, episode_number):
    for entry in feed.entries:
        if get_episode_number(entry) == episode_number:
            return entry
    return None


def get_audio_url(entry):
    for link in entry.get("links", []):
        if link.get("type", "").startswith("audio/"):
            return link["href"]
    for enc in entry.get("enclosures", []):
        if enc.get("type", "").startswith("audio/"):
            return enc["href"]
    return None


def download_audio(url, episode_number):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filepath = AUDIO_DIR / f"episode_{episode_number}.mp3"
    if filepath.exists():
        print(f"  Audio already cached: {filepath}")
        return filepath
    print("  Downloading audio...")
    resp = requests.get(url, stream=True, timeout=30, headers=HTTP_HEADERS)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    return filepath


def transcribe_audio(audio_path, model_name="medium"):
    import whisper  # only needed for --source rss

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"{audio_path.stem}.txt"
    if transcript_path.exists():
        print(f"  Transcript already cached: {transcript_path}")
        return transcript_path.read_text()
    print(f"  Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print("  Transcribing (this may take a while)...")
    text = model.transcribe(str(audio_path), verbose=False)["text"]
    transcript_path.write_text(text)
    return text


def rss_episode(feed, episode_number, whisper_model):
    entry = find_episode(feed, episode_number)
    if not entry:
        raise GenerationError(f"Episode #{episode_number} not found in RSS feed.")
    meta = {"title": entry.get("title", f"Episode {episode_number}"), "date": ""}
    t = getattr(entry, "published_parsed", None)
    if t:
        meta["date"] = f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"
    audio_url = get_audio_url(entry)
    if not audio_url:
        raise GenerationError("No audio URL found for this episode.")
    audio_file = download_audio(audio_url, episode_number)
    return meta, transcribe_audio(audio_file, whisper_model)


# ---------------------------------------------------------------------------
# Claude Code (default engine: runs on your Claude plan, no API key)
# ---------------------------------------------------------------------------

class UsageLimitReached(Exception):
    """Claude Code says the plan's usage limit is used up; stop the whole run."""


LIMIT_RE = re.compile(r"usage limit|limit reached|hit your limit|rate limit|out of (?:extra )?usage", re.I)

# Environment variables that would make Claude Code bill an API account instead
# of using the signed-in Claude plan. They are removed before each call.
API_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "ANTHROPIC_BASE_URL")


def check_claude_code():
    import shutil

    if not shutil.which("claude"):
        sys.exit("Claude Code isn't installed (no `claude` command found). "
                 "Install it, run `claude` once to sign in with your Claude account, then try again.")
    if any(os.environ.get(v) for v in API_ENV_VARS):
        print("Note: API settings found in your environment (like ANTHROPIC_API_KEY). "
              "They are removed for these Claude Code calls, so only your Claude plan is used.")


def extract_topics_claude_code(transcript, episode_number, model):
    """Ask Claude Code (`claude -p`, no tools) for the topic graph."""
    import subprocess
    import tempfile

    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--json-schema", json.dumps(GRAPH_SCHEMA),
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt", "You turn podcast transcripts into topic graphs. Reply only with the requested JSON.",
    ]
    if model:
        cmd += ["--model", model]
    prompt = (
        f"<transcript episode=\"{episode_number}\">\n{transcript}\n</transcript>\n\n{EXTRACTION_PROMPT}"
    )
    env = {k: v for k, v in os.environ.items() if k not in API_ENV_VARS}

    print("  Sending transcript to Claude Code for topic extraction...")
    with tempfile.TemporaryDirectory() as workdir:  # empty folder: no project files or CLAUDE.md
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  env=env, cwd=workdir, timeout=30 * 60)
        except subprocess.TimeoutExpired:
            raise GenerationError("Claude Code took longer than 30 minutes, skipping.")

    out = (proc.stdout or "").strip()
    try:
        result = json.loads(out)
    except json.JSONDecodeError:
        result = None

    if not isinstance(result, dict):
        detail = (proc.stderr or out or "no output").strip()[-400:]
        if LIMIT_RE.search(detail):
            raise UsageLimitReached(detail)
        raise GenerationError(f"Claude Code failed (exit {proc.returncode}): {detail}")

    text = str(result.get("result") or "")
    if result.get("is_error") or proc.returncode != 0:
        detail = (text or result.get("subtype") or proc.stderr or "unknown error").strip()[-400:]
        if LIMIT_RE.search(detail):
            raise UsageLimitReached(detail)
        raise GenerationError(f"Claude Code reported an error: {detail}")

    data = result.get("structured_output")
    if data is None:
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            debug_path = SCRIPT_DIR / f".debug_response_{episode_number}.txt"
            debug_path.write_text(out)
            raise GenerationError(f"Couldn't parse Claude Code's JSON ({e}); output saved to {debug_path.name}")

    nodes = data.get("nodes", [])
    links = data.get("links", [])
    print(f"  Extracted {len(nodes)} topics and {len(links)} connections.")
    return nodes, links


# ---------------------------------------------------------------------------
# Claude API (only with --engine api; billed to an API account)
# ---------------------------------------------------------------------------

def make_client():
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        print("Get one at https://console.anthropic.com/  (or use --dry-run, which needs none)", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(max_retries=5)


def episode_cost(model, usage):
    price_in, price_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    tokens_in = (usage.input_tokens or 0) \
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) \
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
    tokens_out = usage.output_tokens or 0
    return tokens_in, tokens_out, (tokens_in * price_in + tokens_out * price_out) / 1e6


def extract_topics(client, transcript, episode_number, model, effort):
    """Ask Claude for the topic graph. Returns (nodes, links, cost_in_dollars)."""
    import anthropic

    print(f"  Sending transcript to {model} for topic extraction...")
    try:
        with client.beta.messages.stream(
            model=model,
            max_tokens=64000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if the model declines, the API retries on its recommended fallback
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": GRAPH_SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": (
                    f"<transcript episode=\"{episode_number}\">\n{transcript}\n</transcript>\n\n"
                    f"{EXTRACTION_PROMPT}"
                ),
            }],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError:
        print("Error: the API key was rejected. Check ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)
    except anthropic.PermissionDeniedError as e:
        print(f"Error: the API key can't use {model}: {e.message}", file=sys.stderr)
        sys.exit(1)
    except anthropic.NotFoundError as e:
        print(f"Error: model or endpoint not found: {e.message}", file=sys.stderr)
        sys.exit(1)
    except anthropic.RateLimitError as e:
        raise GenerationError(f"Rate limited even after retries: {e.message}")
    except anthropic.APIStatusError as e:
        raise GenerationError(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError as e:
        raise GenerationError(f"Network error talking to the API: {e}")

    tokens_in, tokens_out, cost = episode_cost(model, message.usage)
    print(f"  Tokens: {tokens_in:,} in, {tokens_out:,} out (~${cost:.2f})")

    if message.stop_reason == "refusal":
        raise GenerationError("Claude declined this transcript (refusal), skipping.")
    if message.stop_reason == "max_tokens":
        raise GenerationError("Claude's answer was cut off (max_tokens), skipping.")

    raw = next((b.text for b in message.content if b.type == "text"), "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        debug_path = SCRIPT_DIR / f".debug_response_{episode_number}.txt"
        debug_path.write_text(raw)
        raise GenerationError(f"Couldn't parse Claude's JSON ({e}); raw reply saved to {debug_path.name}")

    nodes = data.get("nodes", [])
    links = data.get("links", [])
    print(f"  Extracted {len(nodes)} topics and {len(links)} connections.")
    return nodes, links, cost


def validate_graph(nodes, links):
    """Drop broken links, fix categories and sizes, remove empty optional fields."""
    seen = set()
    clean_nodes = []
    for node in nodes:
        if not node.get("id") or node["id"] in seen:
            continue
        seen.add(node["id"])
        if node.get("cat") not in CATEGORIES:
            node["cat"] = "topic"
        node["size"] = max(7, min(18, int(node.get("size") or 10)))
        if not str(node.get("url", "")).startswith("http"):
            node.pop("url", None)
        clean_nodes.append(node)

    valid_links = []
    for link in links:
        ok = link.get("source") in seen and link.get("target") in seen and link["source"] != link["target"]
        if ok:
            if not link.get("label"):
                link.pop("label", None)
            valid_links.append(link)
        else:
            print(f"  Warning: dropping link {link.get('source')} -> {link.get('target')}")
    return clean_nodes, valid_links


# ---------------------------------------------------------------------------
# Site output
# ---------------------------------------------------------------------------

def generate_html(nodes, links, episode_number):
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE_PATH.read_text()
    html = template.replace("{{EPISODE_NUMBER}}", str(episode_number))
    html = html.replace("{{NODES_JSON}}", json.dumps(nodes, indent=2, ensure_ascii=False))
    html = html.replace("{{LINKS_JSON}}", json.dumps(links, indent=2, ensure_ascii=False))
    out_path = EPISODES_DIR / f"{episode_number}.html"
    out_path.write_text(html)
    print(f"  Mind map saved: {out_path.relative_to(SCRIPT_DIR)}")


def card_description(nodes):
    top = sorted(nodes, key=lambda n: -n["size"])[:6]
    names = ", ".join(n["label"].replace("\n", " ") for n in top)
    return f"{len(nodes)} topics mapped: {names}, and more."


def update_episodes_json(episode_number, title, date, nodes):
    data = json.loads(EPISODES_JSON.read_text())
    entry = {
        "number": episode_number,
        "title": title,
        "date": date,
        "description": card_description(nodes),
    }
    for i, ep in enumerate(data["episodes"]):
        if ep["number"] == episode_number:
            data["episodes"][i] = entry
            break
    else:
        data["episodes"].append(entry)
    data["episodes"].sort(key=lambda e: e["number"], reverse=True)
    EPISODES_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_episode_args(values):
    """Accept 442, 442 443, or ranges like 400-410."""
    episodes = []
    for value in values:
        m = re.fullmatch(r"(\d+)-(\d+)", value)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            episodes.extend(range(min(a, b), max(a, b) + 1))
        elif value.isdigit():
            episodes.append(int(value))
        else:
            sys.exit(f"Not an episode number or range: {value}")
    return sorted(set(episodes))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Do By Friday topic connection maps from the archive's transcripts."
    )
    parser.add_argument("--episode", "-e", nargs="+",
                        help="Episode number(s) or ranges like 400-410 (default: latest)")
    parser.add_argument("--all", action="store_true", help="Every numbered episode on the archive")
    parser.add_argument("--engine", default="claude-code", choices=["claude-code", "api"],
                        help="claude-code = Claude Code on this Mac, signed in with your Claude plan (default); "
                             "api = the billed Anthropic API (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--yes", action="store_true", help="Confirm an --all run on the billed API")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and check transcripts only; no Claude calls")
    parser.add_argument("--force", action="store_true", help="Remake maps that already exist")
    parser.add_argument("--refresh", action="store_true", help="Re-download transcripts even if cached")
    parser.add_argument("--model", help="Claude model (default: Claude Code's own default; "
                                         f"{DEFAULT_MODEL} with --engine api)")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                        help="How hard Claude thinks, --engine api only (default: medium)")
    parser.add_argument("--source", default="archive", choices=["archive", "rss"],
                        help="archive = transcripts from the DBF Archives site (default); "
                             "rss = download audio and run Whisper")
    parser.add_argument("--transcript", "-t", help="Path to your own transcript file (one episode only)")
    parser.add_argument("--whisper-model", "-w", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"], help="Whisper model for --source rss")
    args = parser.parse_args()
    if args.engine == "api" and not args.model:
        args.model = DEFAULT_MODEL

    if not TEMPLATE_PATH.exists() or not EPISODES_JSON.exists():
        sys.exit("Run this from the dobyfridaymindmaps folder "
                 "(episode_template.html or episodes.json is missing).")

    feed = fetch_feed() if args.source == "rss" else None

    try:
        if args.all:
            if feed is not None:
                episodes = sorted(n for n in (get_episode_number(e) for e in feed.entries) if n is not None)
            else:
                episodes = archive_episode_numbers()
        elif args.episode:
            episodes = parse_episode_args(args.episode)
        elif feed is not None:
            episodes = [get_episode_number(feed.entries[0])]
        else:
            episodes = [archive_episode_numbers()[-1]]
    except GenerationError as e:
        sys.exit(f"Error: {e}")

    if args.transcript and len(episodes) != 1:
        sys.exit("--transcript works with exactly one --episode.")

    if not args.force and not args.dry_run:
        skipped = [n for n in episodes if (EPISODES_DIR / f"{n}.html").exists()]
        episodes = [n for n in episodes if n not in skipped]
        if skipped:
            more = " ..." if len(skipped) > 10 else ""
            print(f"Skipping {len(skipped)} episode(s) that already have a map (use --force to remake): "
                  f"{', '.join(map(str, skipped[:10]))}{more}")

    if not episodes:
        print("Nothing to do.")
        return

    print(f"Episodes to process: {len(episodes)} ({episodes[0]}–{episodes[-1]})")
    if args.engine == "api" and args.all and not args.dry_run and not args.yes:
        price_in, price_out = PRICES.get(args.model, PRICES[DEFAULT_MODEL])
        low = len(episodes) * (12_000 * price_in + 8_000 * price_out) / 1e6
        high = len(episodes) * (20_000 * price_in + 20_000 * price_out) / 1e6
        print(f"That's {len(episodes)} Claude calls on {args.model}, roughly ${low:.0f}–${high:.0f}.")
        print("Nothing was sent. Add --yes to run it, or --dry-run to check the transcripts for free.")
        return

    client = None
    if not args.dry_run:
        if args.engine == "api":
            client = make_client()
        else:
            check_claude_code()
    done, failed, total_cost, streak = 0, [], 0.0, 0

    for n in episodes:
        print(f"\n{'=' * 60}\nEpisode #{n}\n{'=' * 60}")
        try:
            if args.transcript:
                meta = {"title": f"Episode {n}", "date": ""}
                if args.source == "archive":
                    try:
                        meta, _ = fetch_archive_episode(n, args.refresh)
                    except GenerationError:
                        pass
                text = Path(args.transcript).read_text()
                print(f"  Using provided transcript: {args.transcript}")
            elif args.source == "archive":
                meta, paragraphs = fetch_archive_episode(n, args.refresh)
                text = transcript_text(paragraphs)
                print(f"  {n}. {meta['title']} · {meta.get('date', '')}")
                describe_transcript(meta, paragraphs, text)
            else:
                meta, text = rss_episode(feed, n, args.whisper_model)
                print(f"  {meta['title']} · {meta['date']}")

            if args.dry_run:
                done += 1
                streak = 0
                continue

            if args.engine == "api":
                nodes, links, cost = extract_topics(client, text, n, args.model, args.effort)
                total_cost += cost
            else:
                nodes, links = extract_topics_claude_code(text, n, args.model)
            nodes, links = validate_graph(nodes, links)
            if not nodes:
                raise GenerationError("Claude returned no topics.")
            generate_html(nodes, links, n)
            update_episodes_json(n, meta["title"], meta.get("date", ""), nodes)
            done += 1
            streak = 0
        except GenerationError as e:
            print(f"  Failed: {e}", file=sys.stderr)
            failed.append(n)
            streak += 1
            if streak >= 3:
                print("Three episodes in a row failed, so I'm stopping here. Check the errors above.")
                break
        except UsageLimitReached as e:
            print(f"\n  Claude Code says your usage limit is reached: {e}")
            print("Stopped here. Maps made so far are saved. Run the same command again after your")
            print("limit resets; it skips episodes that already have a map.")
            break
        except KeyboardInterrupt:
            print("\nStopped. Finished maps are kept; rerun the same command to pick up where you left off.")
            break

    print(f"\n{'=' * 60}")
    verb = "checked" if args.dry_run else "mapped"
    print(f"Finished: {done} {verb}, {len(failed)} failed{f' ({failed})' if failed else ''}.")
    if not args.dry_run:
        if args.engine == "api":
            print(f"Estimated Claude API cost this run: ${total_cost:.2f}")
        if done:
            print("Preview: python3 -m http.server 8000  →  http://localhost:8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
