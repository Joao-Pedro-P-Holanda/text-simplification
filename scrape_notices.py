"""
This module scrape the complete version of OMITTED university notices
"""

import asyncio
from dotenv import load_dotenv
import re
from bs4 import BeautifulSoup
import os
import httpx
from urllib.parse import urljoin


def _id_for_file(count: int, year: int):
    return f"{year}_OMITTED_{count}"


async def main():
    base_path = os.getenv("BASE_URL")
    years = range(2020, 2026)

    for year in years:
        print(f"Scraping {year}")
        notices_page_html = httpx.get(
            urljoin(base_path, str(year) + "-2"), follow_redirects=True
        ).text
        soup = BeautifulSoup(notices_page_html, features="html.parser")

        links = _selection_for_year(soup, year)

        for i, link in enumerate(links):
            print(f"Downloading file on {link}")
            response = httpx.get(link, timeout=30)
            os.makedirs("./data/complete", exist_ok=True)
            with open(f"./data/complete/{_id_for_file(i + 1, year)}.pdf", "wb") as f:
                f.write(response.content)


def _selection_for_year(soup: BeautifulSoup, year: int) -> list[str]:
    notices_pattern = re.compile(r"Edital", re.IGNORECASE)
    exclude_patterns = [
        re.compile(
            r"simplificad",
            re.IGNORECASE,
        ),
        re.compile(r"simples", re.IGNORECASE),
        re.compile(r"retificad", re.IGNORECASE),
    ]

    links = []
    match year:
        case 2025:
            exceptions = [os.getenv("EXCEPTION_URL")]

        case _:
            exceptions = []

    a_tags = soup.find_all(
        "a",
        string=lambda text: text
        and notices_pattern.search(text)
        and not any([pattern.search(text) for pattern in exclude_patterns]),
    )

    links.extend(a["href"] for a in a_tags if a["href"].endswith(".pdf"))

    return [link for link in links if link not in exceptions]


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
