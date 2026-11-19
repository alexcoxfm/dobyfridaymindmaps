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

`generate_maps.py` reads each episode's transcript from the [Do By Friday Archives](https://dobyfridayarchives.pages.dev), has Claude find the topics and connections, and writes `episodes/{number}.html` plus its `episodes.json` entry.

By default it runs Claude through **Claude Code on your Mac**, signed in with your Claude account, so it needs no API key. Any `ANTHROPIC_API_KEY` in your environment is removed for those calls. When Claude Code reports a usage limit, the script stops; run the same command again later and it carries on, because episodes that already have a map are skipped.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Free check: fetch and parse transcripts, no Claude at all
python generate_maps.py --episode 1 250 472 --dry-run

# Make maps through Claude Code (install it and run `claude` once to sign in)
python generate_maps.py --episode 470-472

# Every episode; rerun after a usage limit to pick up where it stopped
python generate_maps.py --all
```

`--engine api` uses the billed Anthropic API instead (needs `pip install anthropic` and `ANTHROPIC_API_KEY`). Transcripts are cached in `.transcripts/`, and `--force` remakes existing maps.

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
