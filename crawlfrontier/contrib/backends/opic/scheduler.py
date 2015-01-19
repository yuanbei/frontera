"""
The scheduler class computes the optimal refresh frequency for a given set of
pages
"""
import math

import schedulerdb


class WCluster(object):
    """Cluster the w_i values of the pages"""

    @staticmethod
    def clamp(w):
        """Make sure 0 <= w <= 1"""
        return min(1.0, max(0.0, w))

    def __init__(self, n_clusters=100):
        """Initialize clusters

        n_clusters -- Number of clusters to use
        """
        self.n_clusters = n_clusters
        # Page change rates
        self.r = n_clusters*[0.0]
        # Page values
        self.w = [0.5/n_clusters*(2*k + 1) for k in xrange(n_clusters)]
        # Number of pages
        self.N = 0

    def cluster_index(self, w):
        """For the given value 'w', find the cluster index"""
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
    """Given a function 'f' it tries to find three points, xa, xb, xc, such that:
        f(xa) > f(xb)
        f(xc) > f(xc)
        xa < xb < xc

    xmin and xmax are used to set the initial interval to search in.
    The search could fail (if for example the function is monotonic), and so
    the max_iter parameter sets the maximum number of function evaluations to
    try.

    Returns:
       Two triples if success: ((xa, xb, xc), (fa, fb, fc))
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
    """Find the minimum of 'f' using the golden section search algorithm

    Arguments:
        xmin, xmax -- Start search inside this interval
        eps        -- Required precision. Stop when the minimum is found inside
                      an interval with length less than this value.
        verbose    -- If true print convergence results

    Returns:
        (xa, xb, xc), (fa, fb, fc) with:

            f(xa) > f(xb)
            f(xc) > f(xc)
            xa < xb < xc

        And:
            |xc - xa| < eps

        None if failure. This only happens if the search could not be
        initialized, veacuse otherwise the algorithm is assured to converge.
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
        """Initialize with the following arguments:

        f          -- Function to be pre-computed
        xmin, xmax -- Domain for the function: [xmin, xmax]
        N          -- Divide [xmin, xmax] in N subintervals
        """
        self.xmin = xmin
        self.xmax = xmax
        self.delta = float(xmax - xmin)/float(N)

        self._values = []
        for i in xrange(N+1):
            self._values.append(f(xmin + i*self.delta))

    def __call__(self, x):
        """Evaluate function at 'x' using linear interpolation

        If x not in [xmin, xmax] use extrapolation. Use with care.

        Returns:
            The value f(x) of the function at x
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


class SchedulerSolver(object):
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

        Arguments:
            values    -- page values ('w' coefficients, with 0 < w <= 1)
            rates     -- page rates ('r' or 'lambda' coefficients, with r > 0)
            frequency -- average page frequency. The solution obeys the
                         constraint:
                             mean(f) = frequency
            verbose   -- If true print convergence info
        """
        # pre-compute the g-function
        self._g_grid = GridFunction(SchedulerSolver._g_eval, -1.0, -1e-6, 1000)

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


class Scheduler(object):
    """Compute the optimal refresh frequency, with a fixed computational
    cost
    """

    def __init__(self, n_clusters=100, db=None):
        self._db = db or schedulerdb.SQLite()

        self._cluster = WCluster(n_clusters)
        for rate, value in self._db.iter():
            if rate and value:
                self._cluster.add(value, rate)

        # TODO: this should be estimated. It's the crawl rate in Hz.
        # Set to 200 pages/min
        self._crawl_rate = 200.0 / 60.0
        self._solver = SchedulerSolver()
        self._changed = True

    @property
    def crawl_rate(self):
        """Get crawl rate"""
        return self._crawl_rate

    @crawl_rate.setter
    def crawl_rate(self, value):
        """Change crawl rate"""
        self._changed = True
        self._crawl_rate = value

    def set_rate(self, page_id, rate_new):
        rate_old, value = self._db.get(page_id)

        self._changed = (rate_new != rate_old)
        if not self._changed:
            return

        if rate_old is None:
            if value is not None:
                # (None, value) -> (rate_new, value)
                self._db.set(page_id, rate_new, value)
                self._cluster.add(value, rate_new)
            else:
                # (None, None) -> (rate_new, None)
                self._db.add(page_id, rate_new, None)
        else:
            if value is not None:
                # (rate_old, value) -> (rate_new, value)
                self._db.set(page_id, rate_new, value)
                self._cluster.delete(value, rate_old)
                self._cluster.add(value, rate_new)
            else:
                # (rate_old, None) -> (rate_new, None)
                self._db.set(page_id, rate_new, None)

    def set_value(self, page_id, value_new):
        rate, value_old = self._db.get(page_id)

        self._changed = (value_new != value_old)
        if not self._changed:
            return

        if value_old is None:
            if rate is not None:
                # (rate, None) -> (rate, value_new)
                self._db.set(page_id, rate, value_new)
                self._cluster.add(value_new, rate)
            else:
                # (None, None) -> (None, value_new)
                self._db.add(page_id, None, value_new)
        else:
            if rate is not None:
                # (rate, value_old) -> (rate, value_new)
                self._db.set(page_id, rate, value_new)
                self._cluster.delete(value_old, rate)
                self._cluster.add(value_new, rate)
            else:
                # (None, value_old) -> (None, value_new)
                self._db.set(page_id, None, value_new)

    def frequency(self, page_id):
        rate, value = self._db.get(page_id)
        if value is not None and rate is not None:
            """Get the optimal refresh frequency for a page"""
            if self._cluster.N <= 1:
                return self._crawl_rate
            else:
                if self._changed:
                    self._solver.solve(
                        self._cluster.w,
                        self._cluster.r,
                        self.crawl_rate/float(self._cluster.N),
                        self._cluster.N
                    )

                return rate*self._solver.g_opt[
                    self._cluster.cluster_index(value)]
        else:
            return None

    def close(self):
        self._db.close()
