import os

from scrapy.exceptions import NotConfigured
from scrapy.utils.conf import get_config


def get_project_conf(check_configured=False, auth=None, project_id=None):
    """ Gets hs auth and project id,
    from following sources in following order of precedence:
    - default parameter values
    - hworker.bot.hsref
    - environment variables
    - scrapy.cfg files

    in order to allow to use codes that needs HS or dash API,
    either locally or from scrapinghub, correctly configured
    """

    conf = {'project_id': os.environ.get('PROJECT_ID'),
            'auth': (os.environ.get('SHUB_APIKEY') +
                     ':' if os.environ.get('SHUB_APIKEY') else None)}
    try:
        from hworker.bot.hsref import hsref
        conf = {'project_id': hsref.projectid, 'auth': hsref.auth}
    except Exception:
        pass

    cfg = {}
    try:
        cfg = dict(get_config().items('deploy'))
    except:
        pass

    try:
        if conf['project_id'] is None:
            conf['project_id'] = cfg.get('project')
    except:
        pass

    try:
        if conf['auth'] is None:
            username = cfg.get('username')
            conf['auth'] = '%s:' % username if username else None
    except:
        pass

    # override with values given in parameters
    conf['auth'] = auth or conf['auth']
    conf['project_id'] = project_id or conf['project_id']

    if check_configured:
        if conf['auth'] is None:
            raise NotConfigured('Auth key not found. Use either SHUB_APIKEY'
                                ' environment variable or scrapy.cfg files')
        if conf['project_id'] is None:
            raise NotConfigured('Project id not found. Use either PROJECT_ID'
                                ' environment variable or scrapy.cfg files')
    return conf
