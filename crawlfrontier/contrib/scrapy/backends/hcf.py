from crawlfrontier.core import models
from crawlfrontier.contrib.backends.hcf import HcfBackend


class ScrapyHcfBackend(HcfBackend):

    name = 'Scrapy HCF Backend'

    spider_attributes = (
        'hcf_consume_from_frontier',
        'hcf_consume_from_slot',
        'hcf_max_batches',
        'hcf_number_of_slots',
        'hcf_write_slot_prefix',
        'slot_callback',
        'request_from_qdata'
    )

    def frontier_start(self, spider=None, **kwargs):
        self.manager.logger.backend.debug("Used spider %s" % spider)
        """ Override some settings from spider if defined """
        if spider:
            for attr in self.spider_attributes:
                if hasattr(spider, attr):
                    setattr(self, attr, getattr(spider, attr))
                    self.manager.logger.backend.debug(
                        "Set an attribute from spider: %s." % attr)
        super(ScrapyHcfBackend, self).frontier_start(**kwargs)

    def _slot_callback(self, request):
        if hasattr(self, 'slot_callback'):
            result = self.slot_callback(request)
            return self.slot_callback(request)
        return super(ScrapyHcfBackend, self)._slot_callback(request)

    def _request_from_qdata(self, fp, qdata):
        # If we add the method from spider, then we need to convert
        # the result scrapy request to the frontier request.
        if hasattr(self, 'request_from_qdata'):
            req = self.request_from_qdata(fp, qdata)
            return models.Request(url=req.url, meta=req.meta)
        return super(ScrapyHcfBackend, self)._request_from_qdata(fp, qdata)
