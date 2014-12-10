from text import DETAILED, SHORT

try:
    import color
except ImportError:
    pass
else:
    from color import COLORED_EVENTS, COLORED_CONSOLE, COLORED_CONSOLE_MANAGER
    from color import COLORED_CONSOLE_BACKEND, COLORED_CONSOLE_DEBUGGING
