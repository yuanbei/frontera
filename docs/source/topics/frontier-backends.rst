========
Backends
========

Frontier :class:`Backend <crawlfrontier.core.components.Backend>` is where the crawling logic/policies lies.
It’s responsible for receiving all the crawl info and selecting the next pages to be crawled.
It's called by the :class:`FrontierManager <crawlfrontier.core.manager.FrontierManager>` after
:class:`Middleware <crawlfrontier.core.components.Middleware>`, using hooks for
:class:`Request <crawlfrontier.core.models.Request>`
and :class:`Response <crawlfrontier.core.models.Response>` processing according to
:ref:`frontier data flow <frontier-data-flow>`.

Unlike :class:`Middleware`, that can have many different instances activated, only one
:class:`Backend <crawlfrontier.core.components.Backend>` can be used per frontier.

Some backends require, depending on the logic implemented, a persistent storage to manage
:class:`Request <crawlfrontier.core.models.Request>`
and :class:`Response <crawlfrontier.core.models.Response>` objects info.


.. _frontier-activating-backend:

Activating a backend
====================

To activate the frontier middleware component, set it through the :setting:`BACKEND` setting.

Here’s an example::

    BACKEND = 'crawlfrontier.contrib.backends.memory.FIFO'

Keep in mind that some backends may need to be enabled through a particular setting. See
:ref:`each backend documentation <frontier-built-in-backend>` for more info.

.. _frontier-writing-backend:

Writing your own backend
===========================

Writing your own frontier backend is easy. Each :class:`Backend <crawlfrontier.core.components.Backend>` component is a
single Python class inherited from :class:`Component <crawlfrontier.core.components.Component>`.

:class:`FrontierManager <crawlfrontier.core.manager.FrontierManager>` will communicate with active
:class:`Backend <crawlfrontier.core.components.Backend>` through the methods described below.


.. autoclass:: crawlfrontier.core.components.Backend

    **Methods**

    .. automethod:: crawlfrontier.core.components.Backend.frontier_start

        :return: None.

    .. automethod:: crawlfrontier.core.components.Backend.frontier_stop

        :return: None.

    .. automethod:: crawlfrontier.core.components.Backend.add_seeds

        :return: None.

    .. automethod:: crawlfrontier.core.components.Backend.get_next_requests

    .. automethod:: crawlfrontier.core.components.Backend.page_crawled

        :return: None.

    .. automethod:: crawlfrontier.core.components.Backend.request_error

        :return: None.

    **Class Methods**

    .. automethod:: crawlfrontier.core.components.Backend.from_manager





.. _frontier-built-in-backend:

Built-in backend reference
=============================

This page describes all :ref:`each backend documentation <frontier-built-in-backend>` components that come with
Crawl Frontier. For information on how to use them and how to write your own middleware, see the
:ref:`backend usage guide. <frontier-writing-backend>`.

To know the default activated :class:`Backend <crawlfrontier.core.components.Backend>` check the
:setting:`BACKEND` setting.

.. _frontier-backends-basic-algorithms:

Basic algorithms
----------------
Some of the built-in :class:`Backend <crawlfrontier.core.components.Backend>` objects implement basic algorithms as
as `FIFO`_/`LIFO`_ or `DFS`_/`BFS`_ for page visit ordering.

Differences between them will be on storage engine used. For instance,
:class:`memory.FIFO <crawlfrontier.contrib.backends.memory.FIFO>` and
:class:`sqlalchemy.FIFO <crawlfrontier.contrib.backends.sqlalchemy.FIFO>` will use the same logic but with different
storage engines.

.. _frontier-backends-memory:


Memory backends
---------------------

This set of :class:`Backend <crawlfrontier.core.components.Backend>` objects will use an `heapq`_ object as storage for
:ref:`basic algorithms <frontier-backends-basic-algorithms>`.


.. class:: crawlfrontier.contrib.backends.memory.BASE

    Base class for in-memory heapq :class:`Backend <crawlfrontier.core.components.Backend>` objects.

.. class:: crawlfrontier.contrib.backends.memory.FIFO

    In-memory heapq :class:`Backend <crawlfrontier.core.components.Backend>` implementation of `FIFO`_ algorithm.

.. class:: crawlfrontier.contrib.backends.memory.LIFO

    In-memory heapq :class:`Backend <crawlfrontier.core.components.Backend>` implementation of `LIFO`_ algorithm.

.. class:: crawlfrontier.contrib.backends.memory.BFS

    In-memory heapq :class:`Backend <crawlfrontier.core.components.Backend>` implementation of `BFS`_ algorithm.

.. class:: crawlfrontier.contrib.backends.memory.DFS

    In-memory heapq :class:`Backend <crawlfrontier.core.components.Backend>` implementation of `DFS`_ algorithm.

.. class:: crawlfrontier.contrib.backends.memory.RANDOM

    In-memory heapq :class:`Backend <crawlfrontier.core.components.Backend>` implementation of a random selection
    algorithm.

.. _frontier-backends-sqlalchemy:

SQLAlchemy backends
---------------------

This set of :class:`Backend <crawlfrontier.core.components.Backend>` objects will use `SQLAlchemy`_ as storage for
:ref:`basic algorithms <frontier-backends-basic-algorithms>`.

By default it uses an in-memory SQLite database as a storage engine, but `any databases supported by SQLAlchemy`_ can
be used.

:class:`Request <crawlfrontier.core.models.Request>` and :class:`Response <crawlfrontier.core.models.Response>` are
represented by a `declarative sqlalchemy model`_::

    class Page(Base):
        __tablename__ = 'pages'
        __table_args__ = (
            UniqueConstraint('url'),
        )
        class State:
            NOT_CRAWLED = 'NOT CRAWLED'
            QUEUED = 'QUEUED'
            CRAWLED = 'CRAWLED'
            ERROR = 'ERROR'

        url = Column(String(1000), nullable=False)
        fingerprint = Column(String(40), primary_key=True, nullable=False, index=True, unique=True)
        depth = Column(Integer, nullable=False)
        created_at = Column(TIMESTAMP, nullable=False)
        status_code = Column(String(20))
        state = Column(String(10))
        error = Column(String(20))

If you need to create your own models, you can do it by using the :setting:`DEFAULT_MODELS` setting::

    DEFAULT_MODELS = {
        'Page': 'crawlfrontier.contrib.backends.sqlalchemy.models.Page',
    }

This setting uses a dictionary where ``key`` represents the name of the model to define and ``value`` the model to use.
If you want for instance to create a model to represent domains::

    DEFAULT_MODELS = {
        'Page': 'crawlfrontier.contrib.backends.sqlalchemy.models.Page',
        'Domain': 'myproject.backends.sqlalchemy.models.Domain',
    }

Models can be accessed from the Backend dictionary attribute ``models``.

For a complete list of all settings used for sqlalchemy backends check the :doc:`settings <frontier-settings>` section.

.. class:: crawlfrontier.contrib.backends.sqlalchemy.BASE

    Base class for SQLAlchemy :class:`Backend <crawlfrontier.core.components.Backend>` objects.

.. class:: crawlfrontier.contrib.backends.sqlalchemy.FIFO

    SQLAlchemy :class:`Backend <crawlfrontier.core.components.Backend>` implementation of `FIFO`_ algorithm.

.. class:: crawlfrontier.contrib.backends.sqlalchemy.LIFO

    SQLAlchemy :class:`Backend <crawlfrontier.core.components.Backend>` implementation of `LIFO`_ algorithm.

.. class:: crawlfrontier.contrib.backends.sqlalchemy.BFS

    SQLAlchemy :class:`Backend <crawlfrontier.core.components.Backend>` implementation of `BFS`_ algorithm.

.. class:: crawlfrontier.contrib.backends.sqlalchemy.DFS

    SQLAlchemy :class:`Backend <crawlfrontier.core.components.Backend>` implementation of `DFS`_ algorithm.

.. class:: crawlfrontier.contrib.backends.sqlalchemy.RANDOM

    SQLAlchemy :class:`Backend <crawlfrontier.core.components.Backend>` implementation of a random selection
    algorithm.



.. _FIFO: http://en.wikipedia.org/wiki/FIFO
.. _LIFO: http://en.wikipedia.org/wiki/LIFO_(computing)
.. _DFS: http://en.wikipedia.org/wiki/Depth-first_search
.. _BFS: http://en.wikipedia.org/wiki/Breadth-first_search
.. _OrderedDict: https://docs.python.org/2/library/collections.html#collections.OrderedDict
.. _heapq: https://docs.python.org/2/library/heapq.html
.. _SQLAlchemy: http://www.sqlalchemy.org/
.. _any databases supported by SQLAlchemy: http://docs.sqlalchemy.org/en/rel_0_9/dialects/index.html
.. _declarative sqlalchemy model: http://docs.sqlalchemy.org/en/rel_0_9/orm/extensions/declarative.html

OPIC backend
------------
The OPIC backend takes its name from the "Online Page Importance
Computation" algorithm, described in::

    Adaptive On-Line Page Importance Computation 
    Abiteboul S., Preda M., Cobena G. 
    2003

The main idea is that we want to crawl pages which are important, and
this importance depends on which pages links to and which pages are
linked by the current page in a manner similar to PageRank. Implementation is in

.. autoclass:: crawlfrontier.contrib.backends.opic.backend.OpicHitsBackend

For more information about the implementations details read the annex
:doc:`OPIC details <opic-backend>`

.. toctree::
   :maxdepth: 2

   opic-backend

Configuration
~~~~~~~~~~~~~

The constructor has a lot of arguments, but they will automatically
filled correctly from global settings. Apart from the common settings
with other backends this backend uses the following additional
settings:

* BACKEND_OPIC_IN_MEMORY (default False) 

  If True all information will be kept in-memory. This will make it run 
  faster but also will consume more memory and, more importantly, you 
  will not be able to resume the crawl after you shut down the spider.

* BACKEND_OPIC_WORKDIR

  If BACKEND_OPIC_IN_MEMORY is False, then all the state information 
  necessary to resume the crawl will be kept inside this directory. If
  this directory is not set a default one will be generated in the 
  current directory following the pattern:: 
    
      crawl-opic-DYYYY.MM.DD-THH.mm.SS

  Where YYYY is the current year, MM is the month, etc...
    
* BACKEND_TEST

  If True the backend will save some information to inspect after the 
  databases are closed to test the backend performance.

.. _opic-databases:

Persistence
~~~~~~~~~~~

The following SQLite databases are generated inside the
BACKEND_OPIC_WORKDIR:

* graph.sqlite: the graph database. See :ref:`graph-database`.

* freqs.sqlite: the output of the scheduler. How frequently each page
  should be crawled. See :ref:`revisiting-scheduler`.

* hash.sqlite: a hash of the body of each page the last time it was
  visited. It allows to track if a page has changed. See
  :ref:`page-change-estimator`

* hits.sqlite: the HITS score information and additional information
  for the OPIC algorithm. See :ref:`hits-scores-db`.

* pages.sqlite: additional info about pages, like URL and domain.

* scheduler.sqlite: page value and page change rate used by the
  scheduler algorithm. See :ref:`revisiting-scheduler`.

* updates.sqlite: number of updates in a given interval of time. Used
  for page change rate estimation. See  :ref:`page-change-estimator`


Since they are standard SQLite databases they can be accessed using
any tool of your choice (for example `sqlitebrowser
<http://sqlitebrowser.org/>`_) which is useful for debugging or interfacing
with other tools. An example would be accesing the data inside the
databases to compare the precision of OPIC and the power method, like
it's explained in :doc:`opic-precision`.


