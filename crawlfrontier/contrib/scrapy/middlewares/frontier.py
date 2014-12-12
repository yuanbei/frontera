from twisted.internet.error import DNSLookupError, TimeoutError
from twisted.internet.task import LoopingCall
from scrapy.exceptions import NotConfigured, DontCloseSpider
from scrapy.http import Request
from scrapy import signals

from crawlfrontier.contrib.scrapy.manager import ScrapyFrontierManager

# Signals
frontier_download_error = object()

# Defaul values
DEFAULT_FRONTIER_ENABLED = True


class CrawlFrontierSpiderMiddleware(object):

    def __init__(self, crawler, stats):
        self.crawler = crawler
        self.stats = stats

        # Enable check
        if not crawler.settings.get('FRONTIER_ENABLED', DEFAULT_FRONTIER_ENABLED):
            raise NotConfigured

        # Frontier
        frontier_settings = crawler.settings.get('FRONTIER_SETTINGS', None)
        if not frontier_settings:
            raise NotConfigured
        self.frontier = ScrapyFrontierManager(frontier_settings)

        # Signals
        self.crawler.signals.connect(self.spider_closed, signals.spider_closed)
        self.crawler.signals.connect(self.download_error, frontier_download_error)

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler, crawler.stats)

    def spider_closed(self, spider, reason):
        self.frontier.stop()

    def process_start_requests(self, start_requests, spider):
        if not self.frontier.manager.auto_start:
            self.frontier.start(spider=spider)

        # Adding seeds on start
        if start_requests:
            self.frontier.add_seeds(start_requests)

        # Start of requests consuming
        for req in self._get_next_requests(spider):
            yield req

    def process_spider_output(self, response, result, spider):
        links = []
        for element in result:
            if isinstance(element, Request):
                links.append(element)
            else:
                yield element
        self.frontier.page_crawled(scrapy_response=response,
                                   scrapy_links=links)

    def download_error(self, request, exception, spider):
        # TODO: Add more errors...
        error = '?'
        if isinstance(exception, DNSLookupError):
            error = 'DNS_ERROR'
        elif isinstance(exception, TimeoutError):
            error = 'TIMEOUT_ERROR'
        self.frontier.request_error(scrapy_request=request, error=error)

    def _get_next_requests(self, spider):
        """ Get new requests from the manager."""

        while not self.frontier.manager.finished:
            for req in self.frontier.get_next_requests():
                yield req


class CrawlFrontierDownloaderMiddleware(object):
    def __init__(self, crawler):
        self.crawler = crawler

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)

    def process_exception(self, request, exception, spider):
        self.crawler.signals.send_catch_log(signal=frontier_download_error,
                                            request=request,
                                            exception=exception,
                                            spider=spider)
