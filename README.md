# football-match-prediction-nn

> ⚠️ **Project in Progress** — This is an early-stage learning project. The data scraping pipeline is functional, but the neural network prediction model is not yet implemented. Expect breaking changes and incomplete features.

## Overview

A machine learning project for predicting Premier League football match outcomes using neural networks. Currently focused on building a robust data pipeline that scrapes historical fixture data from FBRef.

## Current Status

✅ **Completed:**
- Web scraper using Selenium + BeautifulSoup to extract Premier League fixtures from FBRef
- Bypasses bot detection using `undetected-chromedriver`
- Data pipeline outputs structured CSV files with match details (teams, scores, attendance, venue, referee)

🚧 **In Progress:**
- Neural network model design and implementation
- Feature engineering from raw fixture data
- Training and evaluation pipeline

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scraper
python app/services/fbref_data_scraper.py
```

**Note:** Requires Chrome browser installed. The scraper uses `undetected-chromedriver` to manage the ChromeDriver binary automatically.

## Project Structure

```
app/
├── services/
│   ├── fbref_data_scraper.py  # Main scraper
│   └── notebooks/              # Data exploration notebooks
├── data/                       # Scraped CSV/HTML output
└── main.py                     # Entry point (WIP)
```

## Contributing

This is a personal learning project, but feedback and suggestions are welcome via issues.