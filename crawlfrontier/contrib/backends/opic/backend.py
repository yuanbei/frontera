"""
Frontier backend implementation based on the OPIC algorithm adaptated
to HITS scoring.

OPIC stands for On-Line Page Importance Computation and is described
in:

    Adaptive On-Line Page Importance Computation
    Abiteboul S., Preda M., Cobena G.
    2003

HITS stands for Hyperlink-Induced Topic Search, originally described
in:

    Authoritative sources in a hyperlinked environment
    Kleinber J.
    1998

The basic idea of this backend is to crawl pages with a frequency
proportional to a weighted average of the hub and authority scores.
"""
import datetime
import os
import time

from crawlfrontier import Backend
from crawlfrontier.core.models import Request

from opichits import OpicHits

import graphdb
import freqdb
import pagedb
import hitsdb
import hashdb
import updatesdb
import schedulerdb
import freqest
import pagechange
import scheduler


class OpicHitsBackend(Backend):
    """Frontier backend implementation based on the OPIC algorithm adaptated
    to HITS scoring
    """

    DEFAULT_FREQ = 1.0/(30.0*24*3600.0)  # once a month

    component_name = 'OPIC-HITS Backend'

    def __init__(
            self,
            manager,
            db_graph=None,
            db_freqs=None,
            db_pages=None,
            db_hits=None,
            db_scheduler=None,
            freq_estimator=None,
            change_detector=None,
            test=False
    ):
        """
        :param manager: Frontier manager.
        :param db_graph: Graph database. If None use a new instance of :class:`SQLite <.graphdb.SQLite>`.
        :param db_freqs: Frequency database. If None use a new instance :class:`SQLite <crawlfrontier.contrib.backends.opic.freqdb.SQLite>`.
        :param db_pages: Page database. If None us a new instance of :class:`SQLite <crawlfrontier.contrib.backends.opic.pagedb.SQLite>`.
        :param db_hits: HITS database. If None use a new instance of :class:`SQLite <crawlfrontier.contrib.backends.opic.hitsdb.SQLite>`.
        :param db_scheduler: Page refresh scheduler database. If None use a new instance of :class:`SQLite <crawlfrontier.contrib.backends.opic.schedulerdb.SQLite>`.
        :param freq_estimator: Frequency estimator.
        :param change_detector: Change detector.
        :param test: If True compute h_scores and a_scores prior to closing.

        :type manager: :class:`FrontierManager <crawlfrontier.core.manager.FrontierManager>`
        :type db_graph: :class:`GraphInterface <crawlfrontier.contrib.backends.opic.graphdb.GraphInterface>`
        :type db_freqs: :class:`FreqDBInterface <crawlfrontier.contrib.backends.opic.freqdb.FreqDBInterface>`
        :type db_pages: :class:`PageDBInterface <crawlfrontier.contrib.backends.opic.pagedb.PageDBInterface>`
        :type db_hits: :class:`HitsDBInterface <crawlfrontier.contrib.backends.opic.hitsdb.HitsDBInterface>`
        :type db_scheduler: :class:`SchedulerDBInterface <crawlfrontier.contrib.backends.opic.schedulerdb.SchedulerDBInterface>`
        :type freq_estimator: :class:`FreqEstimatorInterface <crawlfrontier.contrib.backends.opic.schedulerdb.FreqEstimatorInterface>`
        :type change_detector: :class:`FreqEstimatorInterface <crawlfrontier.contrib.backends.opic.pagechange.PageChangeInterface>`
        """
        # Adjacency between pages and links
        self._graph = db_graph or graphdb.SQLite()
        # Additional information (URL, domain)
        self._pages = db_pages or pagedb.SQLite()
        # Implementation of the OPIC algorithm
        self._opic = OpicHits(
            db_graph=self._graph,
            db_scores=db_hits or hitsdb.SQLite(),
            time_window=1000.0
        )

        # Estimation of page change frequency
        self._freqest = freq_estimator or freqest.Simple()
        # Detection of a change inside a page
        self._pagechange = change_detector or pagechange.BodySHA1()
        # Algorithm to schedule pages
        self._scheduler = scheduler.Optimal(rate_value_db=db_scheduler,
                                            freq_db=db_freqs)

        self._test = test
        self._manager = manager

    # FrontierManager interface
    @classmethod
    def from_manager(cls, manager):
        """Create a new backend using scrapy settings."""

        in_memory = manager.settings.get('BACKEND_OPIC_IN_MEMORY', False)
        if not in_memory:
            now = datetime.datetime.utcnow()
            workdir = manager.settings.get(
                'BACKEND_OPIC_WORKDIR',
                'crawl-opic-D{0}.{1:02d}.{2:02d}-T{3:02d}.{4:02d}.{5:02d}'.format(
                    now.year,
                    now.month,
                    now.day,
                    now.hour,
                    now.minute,
                    now.second
                )
            )
            if not os.path.isdir(workdir):
                os.mkdir(workdir)

            db_graph = graphdb.SQLite(
                os.path.join(workdir, 'graph.sqlite')
            )
            db_pages = pagedb.SQLite(
                os.path.join(workdir, 'pages.sqlite')
            )
            db_freqs = freqdb.SQLite(
                os.path.join(workdir, 'freqs.sqlite')
            )
            db_hits = hitsdb.SQLite(
                os.path.join(workdir, 'hits.sqlite')
            )

            db_updates = updatesdb.SQLite(
                os.path.join(workdir, 'updates.sqlite')
            )
            db_hash = hashdb.SQLite(
                os.path.join(workdir, 'hash.sqlite')
            )

            db_scheduler = schedulerdb.SQLite(
                os.path.join(workdir, 'scheduler.sqlite')
            )

            manager.logger.backend.debug(
                'OPIC backend workdir: {0}'.format(workdir))
        else:
            db_graph = None
            db_pages = None
            db_freqs = None
            db_hits = None
            db_updates = None
            db_hash = None
            db_scheduler = None

            manager.logger.backend.debug('OPIC backend workdir: in-memory')

        return cls(manager,
                   db_graph,
                   db_freqs,
                   db_pages,
                   db_hits,
                   db_scheduler,
                   freqest.Simple(
                       db=db_updates,
                       default_freq=OpicHitsBackend.DEFAULT_FREQ),
                   pagechange.BodySHA1(db=db_hash),
                   test=manager.settings.get('BACKEND_TEST', False))

    # FrontierManager interface
    def frontier_start(self):
        pass

    # FrontierManager interface
    def frontier_stop(self):
        if self._test:
            self._h_scores = self.h_scores()
            self._a_scores = self.a_scores()

        self._graph.close()
        self._pages.close()
        self._opic.close()
        self._freqest.close()
        self._scheduler.close()

    # Add pages
    ####################################################################
    def _update_freqest(self, page_fingerprint, body=None):
        """Add estimation of page change rate"""
        # check page changes and frequency estimation
        if body:
            page_status = self._pagechange.update(page_fingerprint, body)
        else:
            page_status = None

        if page_status is None or page_status == pagechange.Status.NEW:
            self._freqest.add(page_fingerprint)
        else:
            self._freqest.refresh(
                page_fingerprint, page_status == pagechange.Status.UPDATED)

        # update frequency estimation in scheduler
        self._scheduler.set_rate(
            page_fingerprint,
            self._freqest.frequency(page_fingerprint))

    def _update_page_value(self, page_fingerprint, value=None):
        if value is None:
            h_score, a_score = self._opic.get_scores(page_fingerprint)
        else:
            a_score = value

        self._scheduler.set_value(page_fingerprint, a_score)

    def _add_new_link(self, link):
        """Add a new node to the graph, if not present

        Returns the fingerprint used to add the link
        """
        fingerprint = link.meta['fingerprint']
        self._graph.add_node(fingerprint)
        self._opic.add_page(fingerprint)
        self._pages.add(fingerprint,
                        pagedb.PageData(link.url, link.meta['domain']['name']))

        self._update_freqest(fingerprint, body=None)
        self._update_page_value(fingerprint)
        return fingerprint

    # FrontierManager interface
    def add_seeds(self, seeds):
        """Start crawling from this seeds"""
        tic = time.clock()

        self._graph.start_batch()
        self._pages.start_batch()

        for seed in seeds:
            self._add_new_link(seed)

        self._graph.end_batch()
        self._pages.end_batch()

        toc = time.clock()
        self._manager.logger.backend.debug(
            'PROFILE ADD_SEEDS time: {0:.2f}'.format(toc - tic))

    # FrontierManager interface
    def page_crawled(self, response, links):
        """Add page info to the graph and reset its score"""

        tic = time.clock()

        page_fingerprint = response.meta['fingerprint']
        page_h, page_a = self._opic.get_scores(page_fingerprint)

        self._pages.start_batch()
        self._graph.start_batch()
        for link in links:
            link_fingerprint = self._add_new_link(link)
            self._graph.add_edge(page_fingerprint, link_fingerprint)
            self._graph.end_batch()
        self._pages.end_batch()

        # mark page to update
        self._opic.mark_update(page_fingerprint)

        self._update_freqest(page_fingerprint, body=response.body)

        toc = time.clock()
        self._manager.logger.backend.debug(
            'PROFILE PAGE_CRAWLED time: {0:.2f}'.format(toc - tic))

    # FrontierManager interface
    def request_error(self, page, error):
        """Remove page from frequency db"""
        self._scheduler.delete(page.meta['fingerprint'])

    # Retrieve pages
    ####################################################################
    def _get_request_from_id(self, page_id):
        page_data = self._pages.get(page_id)
        if page_data:
            result = Request(page_data.url)
            result.meta['fingerprint'] = page_id
            if 'domain' not in result.meta:
                result.meta['domain'] = {}
            result.meta['domain']['name'] = page_data.domain
        else:
            result = None

        return result

    # FrontierManager interface
    def get_next_requests(self, max_n_requests):
        """Retrieve the next pages to be crawled"""
        tic = time.clock()

        h_updated, a_updated = self._opic.update()
        for page_id in a_updated:
            self._update_page_value(page_id)

        # build requests for the best scores, which must be strictly positive
        next_pages = self._scheduler.get_next_pages(max_n_requests)
        next_requests = map(self._get_request_from_id, next_pages)

        toc = time.clock()
        self._manager.logger.backend.debug(
            'PROFILE GET_NEXT_REQUESTS time: {0:.2f}'.format(toc - tic))

        return next_requests

    # Just for testing/debugging
    ####################################################################
    def h_scores(self):
        if self._scores._closed:
            return self._h_scores
        else:
            return {self._pages.get(page_id).url: h_score
                    for page_id, h_score, a_score in self._opic.iscores()}

    def a_scores(self):
        if self._scores._closed:
            return self._a_scores
        else:
            return {self._pages.get(page_id).url: a_score
                    for page_id, h_score, a_score in self._opic.iscores()}
