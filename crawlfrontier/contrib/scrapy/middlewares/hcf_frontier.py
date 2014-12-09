from scrapy.http import Request
from crawlfrontier.contrib.scrapy.middlewares.frontier import CrawlFrontierSpiderMiddleware


class HcfCrawlFrontierSpiderMiddleware(CrawlFrontierSpiderMiddleware):

    def process_spider_output(self, response, result, spider):
        links = []
        for element in result:
            if isinstance(element, Request):
                request = element
                # Additional logic for using hcf
                if request.meta.get('use_hcf', False):
                    if request.method == 'GET':
                        links.append(request)
                    else:
                        self.frontier.logger.manager.error(
                            "'use_hcf' meta key is not supported "
                            "for non GET requests (%s)" % request.url)
                        yield request
                else:
                    yield request
            else:
                yield element
        self.frontier.page_crawled(scrapy_response=response,
                                   scrapy_links=links)
        self._remove_queued_request(response.request)
