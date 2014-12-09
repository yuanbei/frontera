"""
HCF Backend

This crawl-frontier backend uses the HCF backend from hubstorage to retrieve
the new urls to crawl and store back the links extracted.

To activate this backend it needs to be set as BACKEND
in the frontier settings, i.e.:

BACKEND = 'crawlfrontier.contrib.backends.hcf.HcfBackend'

And the following settings are required:

    * HCF_FRONTIER  - Frontier name for writting and default reading.

The next optional settings can be defined:

    * HCF_CONSUME_FROM_SLOT     - Slot from where the spider will read new URLs,
                                    (can be None, then should be redefined with an extension).
    * HCF_CONSUME_FROM_FRONTIER - If set, this is the frontier where batches are read.
                                    Otherwise use the value of HCF_FRONTIER.
    * HCF_WRITE_SLOT_PREFIX     - When generating write slot, prepend the given prefix.
                                    Empty by default.
    * HCF_MAX_PARALLEL_BATCHES  - Number of batches to be process in parallel.
                                    The default is 5.
    * HCF_NUMBER_OF_SLOTS       - This is the number of slots that the middleware will
                                    use to store the new links. The default is 8.
    * HCF_SAVE_BATCH_SIZE       - New links batch size to be flushed to the hubstorage.
                                    The default is 1000.

The next keys can be defined in a Request meta in order to control the behavior
of the HCF backend:

    use_hcf     - If set to True the request will be stored in the HCF.
    hcf_request - Dictionary of parameters to be stored in the HCF
                    with the request fingerprint:

        qdata    Data to be stored along with the fingerprint in the request queue
        p        Priority - lower priority numbers are returned first. The default is 0

The value of 'qdata' parameter could be retrieved later using
``response.meta['hcf_request']['qdata']``.

For local runs, project id and apikey are read from the scrapy.cfg config files.
Alternativelly you can use environment variables:

    * SHUB_APIKEY for setting apikey,
    * PROJECT_ID for setting project id
"""

import hashlib
from collections import defaultdict

from hubstorage import HubstorageClient
from toolbox.utils import get_project_conf

from crawlfrontier import Backend
from crawlfrontier.core import models
from crawlfrontier.exceptions import NotConfigured

DEFAULT_HS_NUMBER_OF_SLOTS = 8
DEFAULT_MAX_PARALLEL_BATCHES = 5
DEFAULT_SAVE_BATCH_SIZE = 1000

class HcfBackend(Backend):

    name = 'HCF Backend'

    def __init__(self, manager):

        self.manager = manager
        settings = manager.settings

        conf = get_project_conf(True)
        self.hs_auth = conf['auth']
        self.hs_projectid = conf['project_id']

        self.hcf_frontier = self._get_config(settings, "HS_FRONTIER")

        self.hcf_consume_from_slot = settings.get("HS_CONSUME_FROM_SLOT")
        self.hcf_consume_from_frontier = settings.get('HCF_CONSUME_FROM_FRONTIER')

        self.hcf_number_of_slots = settings.get("HCF_NUMBER_OF_SLOTS", DEFAULT_HS_NUMBER_OF_SLOTS)

        self.hcf_write_slot_prefix = settings.get('HCF_WRITE_SLOT_PREFIX', '')
        self.hcf_max_parallel_batches = settings.get('HCF_MAX_PARALLEL_BATCHES', DEFAULT_MAX_PARALLEL_BATCHES)
        self.hcf_save_batch_size = settings.get('HCF_SAVE_BATCH_SIZE', DEFAULT_SAVE_BATCH_SIZE)

        self.hsclient = HubstorageClient(auth=self.hs_auth)
        self.project = self.hsclient.get_project(self.hs_projectid)
        self.fclient = self.project.frontier

        # ------------ Internal counters ----------

        """ mapping {slot:set(added to slot links)}:
            doesn't allow to add already discovered links to new batches
            though it's cleared on restart, we can occasionally do it

            FIXME probably need to clear it somehow for continuous run
                1) dictionary with ttl
                2) dictionary with size limit """
        self.discovered_links = defaultdict(set)

        """ mapping {link_fingerprint: set(link batch_ids)}
            used to get batch_ids by link fingerprint """
        self.processing_links = defaultdict(set)

        """ mapping {batch_id: set(processing links)}
            allow to check batch_id processing links queue and
            remove batch_id from processing queue """
        self.processing_batches = defaultdict(set)

        # Used for logs and to control data flushing
        self.new_links_count = 0
        # Used for logs only
        self.total_links_count = 0

    def _get_config(self, settings, key, default=None):
        value = settings.get(key, default)
        if value is None:
            raise NotConfigured('%s not found' % key)
        return value

    @classmethod
    def from_manager(cls, manager):
        return cls(manager)

    def frontier_start(self, *kwargs):

        # Define hcf_consume_from_frontier using spider settings or defaults
        if not self.hcf_consume_from_frontier:
            self.hcf_consume_from_frontier = self.hcf_frontier

        self.manager.logger.backend.debug(
            'Using HCF_FRONTIER=%s' % self.hcf_frontier)
        self.manager.logger.backend.debug(
            'Using HCF_CONSUME_FROM_FRONTIER=%s' %
            self.hcf_consume_from_frontier)


    def add_seeds(self, links):
        """ Initially we need to store initial requests with hub-storage,
            and then get it back with get_next_requests() request.  """

        # Check if we have hcf_consume_from_slot defined
        if not self.hcf_consume_from_slot:
            return []

        # Don't need store initial links (for restart) if we already have data
        if self._check_if_data():
            self.manager.logger.backend.debug(
                'Ignoring seeds: already data in HCF')
            return []

        result = []

        # Or walking by seed links
        for link in links:
            slot = self._slot_callback(link)

            hcf_frontier, hcf_request = self.hcf_request_from_link(link)

            # Add hcf requests using hubstorage client
            self.fclient.add(hcf_frontier, slot, [hcf_request])

            # Count links count and flush data from time to time
            self.new_links_count += 1
            if self.new_links_count == self.hcf_save_batch_size:
                self._flush()

            # Store link to the dict as a simple analogue of dupefilter
            self.discovered_links[slot].add(link.url)

            result.append(models.Request(url=link.url,
                                         meta=link.meta))
        # Flush all seeds data after loading
        self._flush()
        self.manager.logger.backend.debug('Added %s seeds to slot(%s)' %
            (len(result), slot))
        # We need to return Requests list to the frontier manager
        return result

    def _check_if_data(self):
        frnt, slot = self.hcf_consume_from_frontier, self.hcf_consume_from_slot
        if list(self.fclient.read(frnt, slot, 1)):
            return True
        return False

    def get_next_requests(self, max_next_requests):
        """ Get a new batch of links from the HCF."""

        # Check if slot is defined
        if not self.hcf_consume_from_slot:
            return []

        requests, new_batches, num_links = [], 0, 0

        for batch in self._get_next_batches(max_next_requests):

            batch_ids = []
            new_batches += 1
            for fingerprint, qdata in batch['requests']:

                num_links += 1
                batch_ids.append(fingerprint)
                self.add_to_processing_links(fingerprint, batch['id'])

                req = self._request_from_qdata(fingerprint, qdata)
                requests.append(req)

            # Add batch to processing_batches only when we processed all links
            self.processing_batches[batch['id']] = batch_ids

        if num_links:
            self.manager.logger.backend.debug('Read %d new batches from slot(%s)' %
                (new_batches, self.hcf_consume_from_slot))
            self.manager.logger.backend.debug('Read %d new links from slot(%s)' %
                (num_links, self.hcf_consume_from_slot))
        return requests

    def _get_next_batches(self, max_next_requests, with_flush=False):

        # Check if we have some batch stock to process
        if self._max_parallel_batches_limit():
            return

        batch_n, is_batches  = 0, False
        for batch_n, batch in enumerate(
                self.fclient.read(self.hcf_consume_from_frontier,
                                  self.hcf_consume_from_slot,
                                  mincount=max_next_requests), 1):
            is_batches = True

            if self._max_parallel_batches_limit():
                break

            # Skip batches already processing
            if batch['id'] in self.processing_batches:
                continue

            yield batch

        # If no batches at all, trying to flush data with hcf client and repeat
        if not is_batches and not with_flush:
            self._flush()
            self._get_next_batches(max_next_requests, with_flush=True)

    def _max_parallel_batches_limit(self):
        if self.hcf_max_parallel_batches:
            if len(self.processing_batches) >= self.hcf_max_parallel_batches:
                return True
        return False

    def page_crawled(self, page, links):
        for link in links:
            if link.meta.get('use_hcf', False):
                slot = self._slot_callback(link)

                # FIXME should we use hash fingerprints only for check?
                rfp = hashlib.sha1(link.url)
                if rfp not in self.discovered_links[slot]:

                    hcf_frontier = self.frontier_from_link(link)
                    hcf_request = self.hcf_request_from_link(link)
                    self.fclient.add(hcf_frontier, slot, [hcf_request])

                    self.discovered_links[slot].add(rfp)

                    # Flush data for each new links batch of certain size
                    self.new_links_count += 1
                    if self.new_links_count == self.hcf_save_batch_size:
                        self._flush()

        self._remove_request_from_queues(page.request)
        return page

    def request_error(self, request, error):
        self.manager.logger.backend.debug(
            'Page crawled error %s' % request.url)
        self._remove_request_from_queues(request)
        return request

    def frontier_stop(self, reason=None):
        self._flush()
        self.manager.logger.backend.debug(
            "Total links flushed: %d" % self.total_links_count)

        # Close the frontier client in order to make sure that
        # all the new links are stored.
        self.fclient.close()
        self.hsclient.close()

    # ================= Generic functions ==================

    def _request_from_qdata(self, fp, qdata):
        # By default use fp as url if no qdata.url is available.
        url = qdata.get('url', fp)
        return models.Request(url=url)

    def _slot_callback(self, request):
        """Determine to which slot should be saved the request  """

        # Use predefined slot from meta if such
        if 'hcf_slot' in request.meta:
            return request.meta['hcf_slot']

        # Allow to specify the number of slots per-request basis.
        nr_slots = request.meta.get('hcf_number_of_slots',
                                    self.hcf_number_of_slots)
        h = hashlib.md5()
        h.update(request.url)
        digest = h.hexdigest()
        return self.hcf_write_slot_prefix + str(int(digest, 16) % nr_slots)

    # ================= Auxiliary functions =================

    def frontier_from_link(self, link):
        """ Get frontier name for a Frontier link """

        # Use metadata value (if exists) to redefine frontier name
        return link.meta.get('hcf_frontier', self.hcf_frontier)

    def hcf_request_from_link(self, link):
        """ Form HCF Request from a Frontier Link """

        # Use metadata value (if exists) to get request
        hcf_request = link.meta.get('hcf_request', {})

        # Use url as a fingerprint if no predefined fingerprint in the meta
        if 'fp' not in hcf_request:
            hcf_request['fp'] = link.url

        hcf_request.setdefault('qdata', {})

        return hcf_request

    def _remove_request_from_queues(self, request):
        """ Clean processing queues """

        # Use the first url from redirect queue if such
        url = (request.meta['redirect_urls'][0]
                if 'redirect_urls' in request.meta else request.url)

        # Get processing batches list for the url
        batches = self.processing_links.pop(url)

        # Remove the url for each batch from list
        for batch_id in batches:
            self.rm_from_processing_batches(batch_id, url)

    def add_to_processing_links(self, fp, batch_id):
        """ Update a processing link batches list with new batch id
            (It's not common, but the same link can be in several
            processing batches, we need to provide such cases).  """

        # Get current batches list for the link fingerprint,
        # add batch_id to this list & update it
        self.processing_links[fp] = self.processing_links.get(fp, []) + [batch_id]

    def rm_from_processing_batches(self, batch_id, fp):
        """ Remove a fingerprint from batch queue """

        # Remove a link fingerprint from the processing batch
        self.processing_batches[batch_id].remove(fp)

        # If no more processing links in the batch,
        # remove the batch from the processing queue &
        # delete it using a hubstorage frontier client.
        if self.processing_batches[batch_id] == []:
            del self.processing_batches[batch_id]
            self.fclient.delete(self.hcf_consume_from_frontier,
                                self.hcf_consume_from_slot,
                                [batch_id])
            self.manager.logger.backend.debug(
                'Deleted batch %s in slot(%s)' % (
                    batch_id, self.hcf_consume_from_slot))

    def _flush(self):
        """ Save new requests and reset counter """

        self.manager.logger.backend.debug(
            'Flushing %d links' % self.new_links_count)

        # Flush data using hubstorage frontier client
        self.fclient.flush()

        # Move new_links counter value to total_links counter
        self.total_links_count += self.new_links_count
        self.new_links_count = 0
