"""
The scheduler class computes the optimal refresh frequency for a given set of
pages
"""
import math
from abc import ABCMeta, abstractmethod, abstractproperty

import schedulerdb
import freqdb


class SchedulerInterface(object):
    """Interface that must be satisfied by all schedulers"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_rate(self, page_id, rate_new):
        """Set change rate for given page"""
        pass

    @abstractmethod
    def set_value(self, page_id, value_new):
        """Set page value for given page"""
        pass

    @abstractmethod
    def delete(self, page_id):
        """Remove page from scheduler"""
        pass

    @abstractmethod
    def get_next_pages(self, n_pages, filter_out=None):
        """Return next pages to crawl.

        :param int n_pages: (maximum) number of pages to return
        :param iterable filter_out: do not return pages in this set
        """
        pass

    @abstractmethod
    def close(self):
        """Close all databases"""
        pass

    def get_crawl_rate(self):
        pass

    def set_crawl_rate(self, value):
        pass

    """Number of pages per second that can be crawled"""
    crawl_rate = abstractproperty(get_crawl_rate, set_crawl_rate)


class WCluster(object):
    """Cluster the w_i values of the pages"""

    @staticmethod
    def clamp(w):
        """Make sure 0 <= w <= 1"""
        return min(1.0, max(0.0, w))

    def __init__(self, n_clusters=100):
        """Initialize clusters

        :param int n_clusters: Number of clusters to use
        """
        self.n_clusters = n_clusters
        # Page change rates
        self.r = n_clusters*[0.0]
        # Page values
        self.w = [0.5/n_clusters*(2*k + 1) for k in xrange(n_clusters)]
        # Number of pages
        self.N = 0

    def cluster_index(self, w):
        """For the given page value, find the cluster index"""
        return int(math.ceil(w*self.n_clusters)) - 1

    def add(self, page_value, page_rate):
        """Adds page to cluster"""
        w = WCluster.clamp(page_value)
        if w > 0:  # ignore if w=0
            self.r[self.cluster_index(w)] += page_rate
            self.N += 1

    def delete(self, page_value, page_rate):
        """Deletes page from cluster"""
        w = WCluster.clamp(page_value)
        if w > 0:  # ignore if w=0
            self.r[self.cluster_index(w)] -= page_rate
            self.N -= 1

    def __str__(self):
        return "WCluster[{0}]".format(self.n_clusters)

    def __repr__(self):
        return " ".join([str(self.N)] + [
            "{0:.6e}".format(r) for r in self.r]
        )


def bracket_min(f, xmin, xmax, max_iter=400):
    """Given a function it tries to find three points such that the middle one
    is the minimum of the three.

    More exactly it tries to find xa, xb, xc in such a way that::

       xa, xb, xc, such that
       f(xa) > f(xb)
       f(xc) > f(xc)
       xa < xb < xc

    xmin and xmax are used to set the initial interval to search in.
    The search could fail (if for example the function is monotonic), and so
    the max_iter parameter sets the maximum number of function evaluations to
    try.

    :param function f: function to bracket
    :param float xmin: left side of the start interval
    :param float xmax: right side of the start interval

    :returns: a pair of triples -- if success: ((xa, xb, xc), (fa, fb, fc))
              None otherwise.
    """

    xa = xmin
    xc = xmax
    xb = 0.5*(xmin + xmax)
    fa = f(xa)
    fb = f(xb)
    fc = f(xc)

    i = 0
    while True:
        fm = min(fa, fb, fc)
        # We found our points
        if fb == fm:
            return ((xa, xb, xc),
                    (fa, fb, fc))
        # Otherwise, expand the search interval towards the current minimum
        elif fa == fm:
            fb = fa
            xb = xa
            xa -= (xc - xb)
            fa = f(xa)
        else:
            fb = fc
            xb = xc
            xc += (xb - xa)
            fc = f(xc)

        i += 1
        if i >= max_iter:
            # Failure
            return None


def golden_section_search(
        f, xmin, xmax, eps=1e-8, verbose=False):
    """Find the minimum of a function using the golden section search algorithm

    :param function f: function to bracket
    :param float xmin: left side of the start interval
    :param float xmax: right side of the start interval
    :param float eps: required precision. Stop when the minimum is found inside
                      an interval with length less than this value
    :param bool verbose: If true print convergence results

    :returns: a pair of triples -- (xa, xb, xc), (fa, fb, fc) or None if failure

    The returned values obey::

       f(xa) > f(xb)
       f(xc) > f(xc)
       xa < xb < xc

       |xc - xa| < eps

    Failure can happen only if the search could not be initialized, because
    otherwise the algorithm is sure to converge.
    """

    # (1 - R)/R = phi = golden ratio
    R = 0.38197
    S = 1 - R

    # Find starting points to the algorithm
    start = bracket_min(f, xmin, xmax)
    if start is not None:
        ((xa, xb, xc), (fa, fb, fc)) = start
    else:
        return None

    while True:
        # Interval lengths
        #
        # a          b    c
        # +----------+----+
        #      L1      L2
        # \---------------/
        #          L
        L = xc - xa
        L1 = xb - xa
        L2 = xc - xb
        # The next point must be found in the largest interval
        if L2 >= L1:
            # a    b          c
            # +----+---*------+
            #          d
            xd = xa + S*L
            fd = f(xd)
            if fd <= fb:
                xa, xb = xb, xd
                fb = fd
            else:
                xc = xd
        else:
            # a        b    c
            # +--*-----+----+
            #    d
            xd = xa + R*L
            fd = f(xd)
            if fd <= fb:
                xb, xc = xd, xb
                fb = fd
            else:
                xa = xd

        if verbose:
            print "f: {0:.2e} eps: {1:.2e}".format(fb, L)

        if L < eps:
            break

    return ((xa, xb, xc),
            (fa, fb, fc))


class GridFunction(object):
    """Precompute function inside a grid"""

    def __init__(self, f, xmin, xmax, N):
        """
        :param function f: function to be pre-computed
        :param float xmin: domain for the function inside [xmin, xmax]
        :param float xmax: domain for the function inside [xmin, xmax]
        :param int N: divide [xmin, xmax] in N subintervals
        """
        self.xmin = xmin
        self.xmax = xmax
        self.delta = float(xmax - xmin)/float(N)

        self._values = []
        for i in xrange(N+1):
            self._values.append(f(xmin + i*self.delta))

    def __call__(self, x):
        """Evaluate function using linear interpolation

        If x not in [xmin, xmax] use extrapolation. Use with care.

        :param float x: where to evaluate the function
        :returns: float -- value of the function at x
        """
        # i is the index of the subinterval
        #                  x
        # ... +---------+--*------+--------+ ...
        #               i         i+1
        i = min(len(self._values) - 2, max(0, int((x - self.xmin)/self.delta)))

        x1 = self.xmin + i*self.delta
        y1 = self._values[i]
        y2 = self._values[i+1]

        return y1 + (x - x1)/self.delta*(y2 - y1)


class OptimalSolver(object):
    @staticmethod
    def _g_eval(x, eps=1e-8, max_iter=1000):
        """Maximization of the Lagrangian relative to the frequency.

        Solves for y: 1 - (1 + 1/y)e(-1/y) + x = 0
        """

        if x <= -1.0:
            return 0
        elif x >= 0.0:
            return float('Inf')
        else:
            i = 0
            u = 0
            while True:
                v = math.log((1.0 + u)/(1.0 + x))
                if abs(v - u) < eps:
                    break
                i += 1
                if i > max_iter:
                    break

                u = v
            return 1.0/u

    def __init__(self, verbose=False):
        """Initialize the solver.

        :param bool verbose: If True print convergence info
        """

        # pre-compute the g-function
        self._g_grid = GridFunction(OptimalSolver._g_eval, -1.0, -1e-6, 1000)

        # For debugging
        self.verbose = verbose
        self.started = False

    def _g(self, x):
        if x <= -1.0:
            return 0
        elif x >= 0.0:
            return float('Inf')
        else:
            return self._g_grid(x)

    def solve(self, values, rates, frequency, n_pages):
        """Get frequencies that maximize page value per unit of time

        :param list values: a list of page values (float in (0,1])
        :param list rates: a list of page change rates (float > 0)
        :param n_pages: number of pages

        It must be that:

            n_pages >= len(values) == len(rates)
        """
        K = len(values)
        N = n_pages

        if not self.started:
            self.started = True

            self._alpha_min = -N*max(values)
            self._alpha_max = 0

        self.f_opt = K*[0.0]
        self.g_opt = K*[0.0]

        def Q(alpha):
            if alpha >= 0:
                return float('Inf')
            q = -alpha*frequency
            # For each page
            for i in xrange(K):
                w = values[i]  # Page value
                r = rates[i]   # Page change rate
                # The page must have some refresh value
                if r > 0 and w > 0:
                    f = r*self._g(alpha/N/w)
                    if f > 0:
                        q += f*(w*(1.0 - math.exp(-r/f)) + alpha*1.0/N)

                    self.f_opt[i] = f
                    self.g_opt[i] = f/r
                else:
                    self.f_opt[i] = 0.0
                    self.g_opt[i] = 0.0

            return q

        ((xa, xb, xc),
         (fa, fb, fc)) = golden_section_search(
            f=Q,
            xmin=self._alpha_min,
            xmax=self._alpha_max,
            eps=1e-2,
            verbose=self.verbose
        )

        self._alpha_min = xa
        self._alpha_max = xc

        return self.f_opt


class Optimal(SchedulerInterface):
    """Compute the optimal refresh frequency, with a fixed computational
    cost
    """

    def __init__(self, n_clusters=100, rate_value_db=None, freq_db=None):
        """
        :param int n_clusters: number of clusters to use for the approximation
        :param rate_value_db: database to store page_id, rate, value triplets
        :param freq_db: database to store the scheduler solution

        :type rate_value_db:
           :class:`SchedulerDBInterface .schedulerdb.SchedulerDBInterface`
        :type freq_db:
           :class:`FreqDBInterface .freqdb.FreqDBInterface`
        """
        super(Optimal, self).__init__()

        self._rv = rate_value_db or schedulerdb.SQLite()
        self._freqs = freq_db or freqdb.SQLite()

        self._cluster = WCluster(n_clusters)
        for rate, value in self._rv.iter():
            if rate and value:
                self._cluster.add(value, rate)

        # TODO: this should be estimated. It's the crawl rate in Hz.
        # Set to 200 pages/min
        self._crawl_rate = 200.0 / 60.0
        self._solver = OptimalSolver()
        self._changed = 0
        self._updated = True

        # When more than this faction of pages has changed either
        # value or rate then solve again
        self.changed_proportion = 0.05

    @property
    def crawl_rate(self):
        """Get crawl rate"""
        return self._crawl_rate

    @crawl_rate.setter
    def crawl_rate(self, value):
        """Change crawl rate"""
        self._updated = True
        self._crawl_rate = value

    def set_rate(self, page_id, rate_new):
        rate_old, value = self._rv.get(page_id)

        changed = (rate_new != rate_old)
        if not changed:
            return
        self._changed += 1

        if rate_old is None:
            if value is not None:
                # (None, value) -> (rate_new, value)
                self._rv.set(page_id, rate_new, value)
                self._cluster.add(value, rate_new)
                self._freqs.add(page_id, self.frequency(page_id))
            else:
                # (None, None) -> (rate_new, None)
                self._rv.add(page_id, rate_new, None)
        else:
            if value is not None:
                # (rate_old, value) -> (rate_new, value)
                self._rv.set(page_id, rate_new, value)
                self._cluster.delete(value, rate_old)
                self._cluster.add(value, rate_new)
                self._freqs.set(page_id, self.frequency(page_id))
            else:
                # (rate_old, None) -> (rate_new, None)
                self._rv.set(page_id, rate_new, None)

    def set_value(self, page_id, value_new):
        value_new = max(1e-12, value_new)
        rate, value_old = self._rv.get(page_id)

        changed = (value_new != value_old)
        if not changed:
            return
        self._changed += 1

        if value_old is None:
            if rate is not None:
                # (rate, None) -> (rate, value_new)
                self._rv.set(page_id, rate, value_new)
                self._cluster.add(value_new, rate)
                self._freqs.add(page_id, self.frequency(page_id))
            else:
                # (None, None) -> (None, value_new)
                self._rv.add(page_id, None, value_new)
        else:
            if rate is not None:
                # (rate, value_old) -> (rate, value_new)
                self._rv.set(page_id, rate, value_new)
                self._cluster.delete(value_old, rate)
                self._cluster.add(value_new, rate)
                self._freqs.set(page_id, self.frequency(page_id))
            else:
                # (None, value_old) -> (None, value_new)
                self._rv.set(page_id, None, value_new)

    def frequency(self, page_id):
        """Get the optimal refresh frequency for a page"""
        rate, value = self._rv.get(page_id)
        if value is not None and rate is not None:
            if self._cluster.N <= 1:
                return self._crawl_rate
            else:
                if self._changed > (self.changed_proportion*self._cluster.N) or\
                   self._updated:
                    self._solver.solve(
                        self._cluster.w,
                        self._cluster.r,
                        self.crawl_rate/float(self._cluster.N),
                        self._cluster.N
                    )
                self._changed = 0
                self._updated = False
                return rate*self._solver.g_opt[
                    self._cluster.cluster_index(value)]
        else:
            return None

    def get_next_pages(self, n_pages, filter_out=None):
        return self._freqs.get_next_pages(n_pages, filter_out)

    def delete(self, page_id):
        rate, value = self._rv.get(page_id)
        if rate is not None and value is not None:
            self._cluster.delete(value, rate)
            self._rv.delete(page_id)
            self._freqs.delete(page_id)

            self._changed += 1

    def close(self):
        self._rv.close()
        self._freqs.close()


class BestFirst(SchedulerInterface):
    def __init__(self, rate_value_db=None):
        """A BestFirst crawler always return the next page with highest value.

        To be really BestFirst get_next_pages should be called with n_pages=1.
        However, the crawler runs OK if its asked for the next best n_pages.

        :param rate_value_db: database to store page_id, rate, value triplets
        :type rate_value_db:
           :class:`SchedulerDBInterface .schedulerdb.SchedulerDBInterface`
        """
        super(BestFirst, self).__init__()
        self._rv = rate_value_db or schedulerdb.SQLite()

    def set_rate(self, page_id, rate_new):
        pass

    def set_value(self, page_id, value_new):
        rate, value_old = self._rv.get(page_id)
        if value_old is None:
            self._rv.add(page_id, rate, value_new)
        else:
            self._rv.set(page_id, rate, value_new)

    def delete(self, page_id):
        self._rv.delete(page_id)

    def get_next_pages(self, n_pages, filter_out=None):
        return self._rv.get_best_value(
            n_pages,
            filter_out,
            delete=True
        )

    def close(self):
        self._rv.close()

    @property
    def crawl_rate(self):
        pass

    @crawl_rate.setter
    def crawl_rate(self, value):
        pass
