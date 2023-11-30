import scrapy

import json
f = open('/home/azaiez/Documents/Cours/These/European Commission/Programs/scraping/DG_urls.json')
DG_urls = json.load(f)
DG_urls = [item['url'] for item in DG_urls]

def _get_eu_member_name(file_name : str):
    for i, world in enumerate(file_name.split(' ')):
        if 'General' in world :
            start = i+1
        if world == 'with':
            stop = i
    EU_member = " ".join(file_name.split(' ')[start:stop])
    return(EU_member)

class CabinetUrlsSpider(scrapy.Spider):
    name = "DG_meetings"
    start_urls = DG_urls

    def start_requests(self):
        for DG_url in self.start_urls :

            # Scrap DG meetings
            yield scrapy.Request(url = DG_url , callback = self.parse_meetings_page, dont_filter=True)

            # Get predecessors urls
            yield scrapy.Request(url = DG_url, callback = self.get_predecessors_urls, dont_filter=True)






    def get_predecessors_urls(self, response):
            pred_url = response.css('a[href*="listPredecessors"]::attr(href)').get()
            print(pred_url)

            if pred_url:
                if not pred_url.startswith("http://") and not pred_url.startswith("https://"):
                    pred_url = 'https://ec.europa.eu/transparencyinitiative/meetings/' + pred_url
                # Create a request to follow the extracted URL

                yield scrapy.Request(pred_url, callback=self.get_predecessors_meetings)

    def get_predecessors_meetings(self, response):

        urls = response.css('a[href*="meeting.do?host"]::attr(href)').getall()[:-1]

        for url in urls :
            if url:
                yield scrapy.Request(url, callback=self.parse_meetings_page)


    def parse_meetings_page(self,response):
        # Extract the title of the HTML page
        title = response.xpath('//h3/text()').get()
        title = title.strip()
        eu_member = _get_eu_member_name(title)

        if eu_member != "":
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
                yield response.follow(next_page_link, self.parse_meetings_page)