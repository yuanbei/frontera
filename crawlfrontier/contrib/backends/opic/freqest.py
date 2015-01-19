"""
A frequency estimator takes into account how many times a page has been
observed changed or unchanged and what points in time, to estimate the
change rate of the page
"""
import time
from abc import ABCMeta, abstractmethod

import updatesdb


class FreqEstimatorInterface(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(FreqEstimatorInterface, self).__init__()

    @abstractmethod
    def add(self, page_id):
        """Add initial frequency estimation for page_id

        page_id      -- An string which identifies the page
        """
        pass

    @abstractmethod
    def refresh(self, page_id, updated):
        """Add new refresh information for page_id

        updated -- A boolean indicating if the page has changed
        """
        pass

    @abstractmethod
    def delete(self, page_id):
        """Stop tracking page_id"""
        pass

    @abstractmethod
    def frequency(self, page_id):
        """Return the estimated refresh frequency for the page. If the page is
        not being tracked return None"""
        pass

    @abstractmethod
    def close(self):
        """Persist or flush all necesary information"""
        pass


class Simple(FreqEstimatorInterface):
    """
    The simple estimator just computes the frequency as the total
    number of updates divided by the total observation time
    """
    def __init__(self, db=None, clock=None, default_freq=0.0):
        """Initialize estimator

        Arguments:
            db           -- updates database to use. If None provided create a new
                            in-memory one.
            clock        -- A function that returns elapsed time in seconds from a
                            fixed origin in time.
            default_freq -- Return this frequency if not enough info available to
                            compute an estimate
        """
        super(Simple, self).__init__()

        self._db = db or updatesdb.SQLite()
        self._clock = clock or time.time
        self._default_freq = default_freq

    def add(self, page_id):
        at_time = self._clock()
        self._db.add(
            page_id,
            at_time,
            at_time,
            0
        )

    def refresh(self, page_id, updated):
        self._db.increment(page_id, self._clock(), 1 if updated else 0)

    def delete(self, page_id):
        self._db.delete(page_id)

    def frequency(self, page_id):
        (start_time, end_time, updates) = \
            self._db.get(page_id) or (None, None, None)

        if start_time is not None:
            if start_time < end_time:
                return float(updates)/(end_time - start_time)
            else:
                return self._default_freq
        else:
            return None

    def close(self):
        self._db.close()
