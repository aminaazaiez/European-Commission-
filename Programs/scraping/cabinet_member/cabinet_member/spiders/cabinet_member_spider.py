import scrapy
from scrapy.exporters import JsonLinesItemExporter

import json
f = open('/home/azaiez/Documents/Cours/These/European Commission/Programs/scraping/cab_urls.json')
urls = json.load(f)
urls = [item['url'] for item in urls]

# Initialize a list to store items
items = []

class TableSpider(scrapy.Spider):
    name = "table_spider"
    start_urls = urls
    def parse(self, response):
        # Extract the title of the HTML page
        title = response.xpath('//h3/text()').get()
        title = title.strip()

        # Locate the table rows within the tbody element
        rows = response.xpath('//table[@id="listMeetingsTable"]/tbody/tr')

        for row in rows:
            # Extract data from each cell within the row
            eu_members = row.xpath('td[1]//text()').getall()
            date = row.xpath('td[2]/text()').get()
            location = row.xpath('td[3]/text()').get()
            entities = row.xpath('td[4]//text()').getall()
            subjects = row.xpath('td[5]//text()').getall()

            # Clean and format the extracted data
            eu_members = [ eu_member.strip() for eu_member in eu_members if eu_member.strip()]
            date = date.strip() if date else None
            location = location.strip() if location else None
            entities = [entity.strip().split('\t')[0] for entity in entities if entity.strip()]
            subjects = [subject.strip().split('\t')[0] for subject in subjects if subject.strip()]

            # Create a dictionary for the item
            yield {
                "EU Member" : eu_members,
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
        #     path = '/home/azaiez/Documents/Cours/These/European Commission/data/meetings/Cabinets/'
        #     filename = f'{title}.json'
        #     with open( path + filename, 'wb') as file:
        #         exporter = JsonLinesItemExporter(file)
        #         exporter.start_exporting()
        #         for item in items:
        #             exporter.export_item(item)
        #         exporter.finish_exporting()
