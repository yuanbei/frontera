# -*- coding: utf-8 -*-
from urlparse import urlparse
from crawlfrontier.contrib.canonicalsolvers.basic import BasicCanonicalSolver
from crawlfrontier.contrib.backends.hbase import _state

class CrawlStrategy(object):
    def __init__(self):
        self.canonicalsolver = BasicCanonicalSolver()

    def add_seeds(self, seeds):
        scores = {}
        for seed in seeds:
            if seed.meta['state'] is None:
                url, fingerprint, _ = self.canonicalsolver.get_canonical_url(seed)
                scores[fingerprint] = 1.0
                seed.meta['state'] = _state.get_id('QUEUED')
        return scores

    def page_crawled(self, response, links):
        scores = {}
        response.meta['state'] = _state.get_id('CRAWLED')
        if response.meta['score'] < 0.4:
            return {}
        _, response_netloc, _, _, _, _ = urlparse(response.url)
        for link in links:
            if link.meta['state'] is None:
                url, fingerprint, _ = self.canonicalsolver.get_canonical_url(link)
                _, netloc, _, _, _, _ = urlparse(url)
                if response_netloc == netloc:
                    scores[fingerprint] = response.meta['score'] / 2
                    link.meta['state'] = _state.get_id('QUEUED')
                else:
                    scores[fingerprint] = None
                    link.meta['state'] = _state.get_id('NOT_CRAWLED')
        return scores

    def page_error(self, request, error):
        url, fingerprint, _ = self.canonicalsolver.get_canonical_url(request)
        request.meta['state'] = _state.get_id('ERROR')
        return {fingerprint: 0.0}

    def finished(self):
        return False

    def get_score(self, url):
        return 1.0 / len(url)