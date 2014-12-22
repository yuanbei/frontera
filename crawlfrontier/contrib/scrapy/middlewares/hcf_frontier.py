from scrapy.http import Request

from crawlfrontier.contrib.scrapy.manager import RequestConversor
from crawlfrontier.contrib.scrapy.middlewares.frontier import CrawlFrontierSpiderMiddleware 

class HCFCrawlFrontierSpiderMiddleware(CrawlFrontierSpiderMiddleware):

    def process_start_requests(self, start_requests, spider):
        if not self.frontier.manager.auto_start:
            self.frontier.start(spider=spider)

        # Start of requests consuming
        for req in self._get_next_requests(spider):
            yield req

        # If there're no links in the hcf or spider does not want to use it,
        # -> use the start_requests.
        if not self.has_new_requests:
            for req in start_requests:
                frontier_req = RequestConversor.scrapy_to_frontier(req)
                yield RequestConversor.frontier_to_scrapy(frontier_req)

    def process_spider_output(self, response, result, spider):
        links = []
        for element in result:
            if isinstance(element, Request):
                if element.meta.get('use_hcf', False):
                    links.append(element)
                else:
                    yield element
            else:
                yield element
        self.frontier.page_crawled(scrapy_response=response,
                                   scrapy_links=links)
