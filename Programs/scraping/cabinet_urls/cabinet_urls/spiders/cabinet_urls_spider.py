import scrapy

import json
f = open('/home/azaiez/Documents/Cours/These/European Commission/Programs/scraping/com_urls.json')
com_urls = json.load(f)
com_urls = [item['url'] for item in com_urls]



class CabinetUrlsSpider(scrapy.Spider):
    name = "cabinet_urls"
    start_urls = com_urls

    def parse(self, response):
        url = response.css('a[href*="meeting.do?host="]::attr(href)').get()

        if url:
            yield {"url" : url}
            # if '/transparencyinitiative/meetings/' not in url:
            #
            #     yield {"url": 'http://ec.europa.eu/transparencyinitiative/meetings/' + url}
            # else :
            #     yield {"url" : 'http://ec.europa.eu' + url}