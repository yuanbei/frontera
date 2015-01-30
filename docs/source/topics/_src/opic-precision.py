
#!/usr/bin/env python
"""
This script makes a plot comparing the hub/authority score of the OPIC
algorithm against an equivalent power method.

Requirements: 
    - numpy
    - scipy
    - matplotlib
    - crawlfrontier
"""

usage = """Compares the precision of the Power Method against OPIC computing
the HITS score.

python opic-precision.py backend_opic_workdir

The main output is a plot of the page scores using both methods
"""


import sys
import os

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse

from crawlfrontier.contrib.backends.opic.opichits import OpicHits
import crawlfrontier.contrib.backends.opic.graphdb as graphdb
import crawlfrontier.contrib.backends.opic.hitsdb as hitsdb


def graph_to_hits_matrix(db):
    """
    Returns a tuple with:

    i2n: mapping between row/column number and node identifier
    H  : hub matrix
    A  : authority matrix
    h  : virtual page hub vector
    a  : virtual page authority vector
    """
    i2n = [n for n in db.inodes()]
    n2i = {n: i for i, n in enumerate(i2n)}
    N = len(i2n)

    H = scipy.sparse.dok_matrix((N, N), dtype=float)
    A = scipy.sparse.dok_matrix((N, N), dtype=float)
    h = np.zeros((N,))
    a = np.zeros((N,))
    for node in i2n:
        i = n2i[node]
        succ = db.successors(node)
        for s in succ:
            w = 1.0/(len(succ) + 1)
            A[n2i[s], i] = w
            a[i] = w
        pred = db.predecessors(node)
        for p in pred:
            w = 1.0/(len(pred) + 1)
            H[n2i[p], i] = w
            h[i] = w

    return (i2n, H, A, h, a)


def hits_pm(H, A, h, a, err_max=1e-4, iter_max=500, verbose=False):
    """Return hub and authority scores. 
    
    Parameters:

    H  : hub matrix
    A  : authority matrix
    h  : virtual page hub vector
    a  : virtual page authority vector    
    """
    # Number of pages
    N = H.shape[0]

    # Hub score
    x = np.random.rand(N)
    x /= np.sum(x)

    # Authority score
    y = np.random.rand(N)
    y /= np.sum(y)

    vh = 0.0
    va = 0.0

    i = 0
    while True:
        xn = H.dot(y) + va/N
        yn = A.dot(x) + vh/N

        vh = h.dot(y)
        va = a.dot(x)

        sa = np.sum(yn)
        sh = np.sum(xn)
        va /= sa
        yn /= sa
        vh /= sh
        xn /= sh

        eps = max(
            np.linalg.norm(xn - x, ord=np.inf),
            np.linalg.norm(yn - y, ord=np.inf)
        )

        x = xn
        y = yn

        if verbose:
            print "iter={0:06d} eps={1:e}".format(i, eps)

        if eps < err_max:
            break

        i += 1
        if i > iter_max:
            break

    return (x, y)

def precision_crawl(workdir):
    # Load databases inside workdir
    opic1 = OpicHits(
        db_graph=graphdb.SQLite(os.path.join(workdir, 'graph.sqlite')),
        db_scores=hitsdb.SQLite(os.path.join(workdir, 'hits.sqlite'))
    )

    print "Converting crawled graph to sparse matrix... ",
    i2n, H, A, h, a = graph_to_hits_matrix(opic1._graph)
    print "done"

    print "Computing HITS scores using power method... ",
    # h1: hub score, power method
    # a1: authority score, power method
    h_pm, a_pm = hits_pm(H, A, h, a, verbose=False)
    print "done"

    # Estimate error of OPIC
    # ----------------------------------------------

    # Matched against the same pages as power method.
    # h2: hub score, OPIC
    # a2: authority score, OPIC
    h_iter_1, a_iter_1 = zip(*[opic1.get_scores(page_id)
                               for page_id in i2n])

    h_iter_2 = H.dot(A.dot(h_iter_1))
    h_iter_2 /= np.sum(h_iter_2)

    # To compute the error of opic authority score
    a_iter_2 = A.dot(H.dot(a_iter_1))
    a_iter_2 /= np.sum(a_iter_2)

    print "Error of OPIC algorithm (L^inf metric):"
    print "    Hub score      : ", \
          np.linalg.norm(h_iter_2 - h_iter_1, ord=np.inf)
    print "    Authority score: ", \
          np.linalg.norm(a_iter_2 - a_iter_1, ord=np.inf)

    # Compare OPIC against PM
    # ----------------------------------------------

    # h_dist_pm: ordered from lowest to highest hub scores for PM
    # h_opic   : OPIC hub scores following PM page order
    h_dist_pm, h_pm_ids = zip(*sorted(zip(h_pm, i2n)))
    h_opic = [opic1.get_scores(page_id)[0]
              for page_id in h_pm_ids]

    # a_dist_pm: ordered from lowest to highest authority scores for PM
    # a_opic   : OPIC authority scores following PM page order
    a_dist_pm, a_pm_ids = zip(*sorted(zip(a_pm, i2n)))
    a_opic = [opic1.get_scores(page_id)[1]
              for page_id in a_pm_ids]

    h_dist_opic = sorted(h_opic)
    a_dist_opic = sorted(a_opic)

    # Improve OPIC scores
    # ----------------------------------------------
    print "Additional opic iterations"
    opic2 = OpicHits(db_graph=opic1._graph, db_scores=None)
    for i in xrange(10):
        opic2.update(n_iter=1000)
        print "    ", (i+1)*1000

    h_opic_improved = [opic2.get_scores(page_id)[0] for page_id in h_pm_ids]
    a_opic_improved = [opic2.get_scores(page_id)[1] for page_id in a_pm_ids]

    h_dist_opic_improved = sorted(h_opic_improved)
    a_dist_opic_improved = sorted(a_opic_improved)

    # Plot figure
    # ----------------------------------------------
    fig = plt.figure()
    fig.suptitle('Power method vs OPIC')

    # Hub score PM vs OPIC
    ax1 = plt.subplot(221)
    plt.hold(True)
    ax1.set_ylabel('Hub score')
    ax1.set_title('Power method vs OPIC')
    p1, = plt.plot(h_dist_pm, 'r-')
    p2, = plt.plot(h_dist_opic, 'b-')
    p3, = plt.plot(h_opic, 'b.')
    ax1.legend(
        [p1, p2, p3],
        ['Power method', 'OPIC (dist)', 'OPIC'],
        loc='upper left'
    )

    # Authority score PM vs OPIC
    ax2 = plt.subplot(223, sharex=ax1)
    plt.hold(True)
    ax2.set_ylabel('Authority score')
    ax2.set_xlabel('Page')

    p1, = plt.plot(a_dist_pm, 'r-')
    p2, = plt.plot(a_dist_opic, 'b-')
    p3, = plt.plot(a_opic, 'b.')

    # Hub score PM vs OPIC improved
    ax3 = plt.subplot(222, sharex=ax1, sharey=ax1)
    plt.hold(True)
    ax3.set_title('Power method vs OPIC improved')

    p1, = plt.plot(h_dist_pm, 'r-')
    p2, = plt.plot(h_dist_opic_improved, 'b-')
    p3, = plt.plot(h_opic_improved, 'b.')

    # Authority score PM vs OPIC improved
    ax4 = plt.subplot(224, sharex=ax1, sharey=ax2)
    plt.hold(True)
    ax4.set_xlabel('Page')

    p1, = plt.plot(a_dist_pm, 'r-')
    p2, = plt.plot(a_dist_opic_improved, 'b-')
    p3, = plt.plot(a_opic_improved, 'b.')

    return fig


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(usage)

    fig = precision_crawl(sys.argv[1])
    plt.show()
