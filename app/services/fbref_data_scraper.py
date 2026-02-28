import logging
import time
from pathlib import Path

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path(__file__).parent.parent / "data"


class FbrefDataScraper:
    def __init__(self):
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        self.driver = uc.Chrome(options=options, version_main=145)

    def _wait_for_cloudflare(self, timeout: int = 60):
        """Wait for Cloudflare challenge to resolve before proceeding."""
        logger.info("Waiting for Cloudflare challenge to resolve...")
        end_time = time.time() + timeout
        while time.time() < end_time:
            title = self.driver.title.lower()
            if "momento" not in title and "just a moment" not in title:
                logger.info("Cloudflare challenge passed")
                return
            time.sleep(2)
        raise TimeoutError("Cloudflare challenge did not resolve within timeout")

    def get_data(self, season: str = "2024-2025") -> pd.DataFrame:
        url = (
            f"https://fbref.com/en/comps/9/{season}/schedule/"
            f"{season}-Premier-League-Scores-and-Fixtures"
        )
        logger.info(f"Navigating to {url}")
        table_id = f"sched_{season}_9_1"
        self.driver.get(url)

        self._wait_for_cloudflare()

        # Wait until the fixtures table is present using its ID
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.XPATH, f'//*[@id="{table_id}"]'))
        )

        # Get fully-rendered outerHTML from the live DOM via JavaScript
        table_element = self.driver.find_element(
            By.XPATH, f'//*[@id="{table_id}"]'
        )
        table_html = self.driver.execute_script(
            "return arguments[0].outerHTML;", table_element
        )

        # Parse with BeautifulSoup using the exact data-stat attribute names
        # from fbref HTML
        soup = BeautifulSoup(table_html, "lxml")
        base = "https://fbref.com"
        rows = []
        for tr in soup.select("tr[data-row]"):
            # Skip spacer rows (no td's with actual data)
            if not tr.find("td", {"data-stat": "date"}):
                continue

            def txt(tag):
                return tag.get_text(strip=True) if tag else None

            attendance_raw = txt(tr.find("td", {"data-stat": "attendance"}))

            mr_td = tr.find("td", {"data-stat": "match_report"})
            mr_a = mr_td.find("a") if mr_td else None
            match_report_link = (
                base + mr_a["href"]
                if mr_a and mr_a.has_attr("href")
                else None
            )

            row = {
                "week": txt(tr.find("th", {"data-stat": "gameweek"})),
                "day": txt(tr.find("td", {"data-stat": "dayofweek"})),
                "date": txt(tr.find("td", {"data-stat": "date"})),
                "time": txt(tr.find("td", {"data-stat": "start_time"})),
                "home": txt(tr.find("td", {"data-stat": "home_team"})),
                "score": txt(tr.find("td", {"data-stat": "score"})),
                "away": txt(tr.find("td", {"data-stat": "away_team"})),
                "attendance": int(attendance_raw.replace(",", "")) if attendance_raw else None,
                "venue": txt(tr.find("td", {"data-stat": "venue"})),
                "referee": txt(tr.find("td", {"data-stat": "referee"})),
                "match_report": match_report_link,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Found fixtures table with shape: {df.shape}")
        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str = "fixtures.csv") -> Path:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return output_path

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None


if __name__ == "__main__":
    fbref_data_scraper = FbrefDataScraper()
    try:
        season = "2023-2024"  # You can change this to scrape different seasons
        df = fbref_data_scraper.get_data(season=season)
        logger.info(f"\n{df.head()}")
        logger.info(f"\nShape: {df.shape}")
        fbref_data_scraper.save_to_csv(df, filename=f"fixtures_{season}.csv")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        fbref_data_scraper.close()


# TODO
# 1. Think about how make the prediction
# 2. make the current code useful for different years
# Find where is the error in the code and fix it
