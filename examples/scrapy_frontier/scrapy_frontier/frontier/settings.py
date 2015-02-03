#--------------------------------------------------------
# Frontier
#--------------------------------------------------------
BACKEND = 'crawlfrontier.contrib.backends.opic.backend.OpicHitsBackend'
BACKEND_OPIC_WORKDIR = 'crawl-opic'
BACKEND_OPIC_SCHEDULER = 'optimal'
MAX_REQUESTS = 2000000
MAX_NEXT_REQUESTS = 10

#--------------------------------------------------------
# Logging
#--------------------------------------------------------
LOGGING_EVENTS_ENABLED = True
LOGGING_MANAGER_ENABLED = True
LOGGING_BACKEND_ENABLED = True
LOGGING_DEBUGGING_ENABLED = False
