"""pyiga

A Python research toolbox for Isogeometric Analysis.
"""

__version__ = '0.1.0'

_max_threads = None

def get_max_threads():
    global _max_threads
    if not _max_threads:
        import multiprocessing
        _max_threads = multiprocessing.cpu_count()
    return _max_threads

def set_max_threads(num):
    global _max_threads
    _max_threads = num
