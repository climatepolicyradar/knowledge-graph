"""
Scrapy spider to scrape the Green Climate Fund website for documents.

Usage:
poetry run scrapy runspider scripts/sampling_for_sectors_classifier/scrape_green_climate_funds.py -O data/raw/green-climate-fund.json --loglevel=INFO
"""

import scrapy

from src.identifiers import generate_identifier


class GreenClimateFundsSpider(scrapy.Spider):
    """
    Scraper for the Green Climate Fund website

    GCFs represent part of our future Multilateral Climate Funds (MCFs) dataset.
    """

    name = "green_climate_funds"
    start_urls = ["https://www.greenclimate.fund/publications/documents"]
    n_pages = 25

    def __init__(self, *args, **kwargs):
        super(GreenClimateFundsSpider, self).__init__(*args, **kwargs)
        self.n_pages = kwargs.get("n_pages", self.n_pages)

    def parse(self, response):  # noqa D102
        for document in response.css("tbody tr"):
            # follow the link to the document page
            document_page_url = document.css("td.views-field-title a::attr(href)").get()
            if document_page_url:
                yield response.follow(document_page_url, self.parse_document)

        # follow pagination links
        next_page = response.css("li.pager-next a::attr(href)").get()
        if next_page is not None and self.n_pages > 0:
            self.n_pages -= 1
            yield response.follow(next_page, self.parse)

    def parse_document(self, response):  # noqa D102
        pdf_url = response.css("a[title='Download']::attr(href)").get().strip()
        title = (
            response.css("h1.d-none.d-md-block.h4.text-primary.mt-0.mb-7::text")
            .get()
            .strip()
        )

        country = (
            response.css("div.field-name-field-country .field-item::text").get().strip()
        )

        if pdf_url and title and country:
            yield {
                "id": generate_identifier(input_string=title + pdf_url),
                "pdf_url": pdf_url,
                "title": title,
                "country": country,
            }

        else:
            self.logger.warning(
                f"Skipping document with missing information: {pdf_url}"
            )
