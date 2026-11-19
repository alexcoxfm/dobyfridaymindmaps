# Do By Friday - Topic Connection Maps

Interactive force-directed graphs showing how topics, people, movies, music, and ideas connect within each episode of [Do By Friday](https://dobyfriday.com).

Built with [D3.js](https://d3js.org). Hosted on GitHub Pages.

## How to add a new episode

1. Create the episode's HTML file (self-contained D3 force-directed graph)
2. Save it as `episodes/{number}.html`
3. Add an entry to `episodes.json`:

```json
{
  "number": 442,
  "title": "Episode Title",
  "date": "2026-03-07",
  "description": "Brief description for the card."
}
```

4. Commit and push. The landing page picks it up automatically.

## Making maps automatically

`generate_maps.py` reads each episode's transcript from the [Do By Friday Archives](https://dobyfridayarchives.pages.dev), asks Claude for the topics and connections, and writes `episodes/{number}.html` plus its `episodes.json` entry.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Free check: fetch and parse transcripts, no API key needed
python generate_maps.py --episode 1 250 472 --dry-run

# Make maps (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python generate_maps.py --episode 470-472

# Every episode: prints the plan and a cost estimate; add --yes to run it
python generate_maps.py --all
```

Episodes that already have a map are skipped unless you pass `--force`. Transcripts are cached in `.transcripts/`. `--model claude-sonnet-5-5` costs about half as much as the default.

## Local preview

The landing page uses `fetch()` to load `episodes.json`, so it needs to be served over HTTP:

```bash
python3 -m http.server 8000
# Open http://localhost:8000
```

## Project structure

```
index.html          Landing page (fetches episodes.json, renders episode cards)
episodes.json       Episode registry (the one file you update per episode)
episodes/           Self-contained episode map HTML files
css/style.css       Landing page styles
404.html            Custom 404 page
.nojekyll           Tells GitHub Pages to skip Jekyll
```
