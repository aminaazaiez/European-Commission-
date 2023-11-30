import scrapy
from scrapy.exporters import JsonLinesItemExporter

import json
f = open('/home/azaiez/Documents/Cours/These/European Commission/Programs/scraping/com_urls.json')
com_urls = json.load(f)
com_urls = [item['url'] for item in com_urls]


def _get_eu_member_name(file_name : str):
    for i, world in enumerate(file_name.split(' ')):
        if 'President' in world :
            start = i+1
        elif 'Commissioner' in world :
            start = i+1
        elif 'Representative' in world :
            start = i+1
        if world == 'with':
            stop = i
    EU_member = " ".join(file_name.split(' ')[start:stop])
    return(EU_member)

# # Initialize a list to store items
# items = []

class TableSpider(scrapy.Spider):
    name = "table_spider"
    start_urls = com_urls


    def parse(self, response):
        # Extract the title of the HTML page
        title = response.xpath('//h3/text()').get()
        title = title.strip()
        eu_member = _get_eu_member_name(title)

        # Locate the table rows within the tbody element
        rows = response.xpath('//table[@id="listMeetingsTable"]/tbody/tr')

        for row in rows:
            # Extract data from each cell within the row
            date = row.xpath('td[1]/text()').get()
            location = row.xpath('td[2]/text()').get()
            entities = row.xpath('td[3]//text()').getall()
            subjects = row.xpath('td[4]//text()').getall()

            # Clean and format the extracted data
            date = date.strip() if date else None
            location = location.strip() if location else None
            entities = [entity.strip().split('\t')[0] for entity in entities if entity.strip()]
            subjects = [subject.strip().split('\t')[0] for subject in subjects if subject.strip()]

            # Create a dictionary for the item
            yield {
                "EU Member" : [eu_member],
                "Date": date,
                "Location": location,
                "Entities": entities,
                "Subjects": subjects,
            }

            # # Append the item to the list
            # items.append(item)

        # Follow the "Next" link if available
        next_page_link = response.css('img[alt="Next"]').xpath('../@href').get()
        if next_page_link:
            yield response.follow(next_page_link, self.parse)
        # else:
        #     # Save all items to a JSON file
        #     path = '/home/azaiez/Documents/Cours/These/European Commission/data/meetings/Commissioners/'
        #     filename = f'{title}.json'
        #     with open(path + filename, 'wb') as file:
        #         exporter = JsonLinesItemExporter(file)
        #         exporter.start_exporting()
        #         for item in items:
        #             exporter.export_item(item)
        #         exporter.finish_exporting()

