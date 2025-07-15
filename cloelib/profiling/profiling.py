"""
Profiling utilities for cloe-lib.

Provides decorators and helpers for runtime profiling using pyinstrument.
"""

import os
import functools
from pyinstrument import Profiler
import datetime


PROFILE_OUTPUT_DIR = "profiling_results"
PROFILER_INTERVAL = 0.001

#class to avoid multiple call of the profiling in case of decorated nested fuctions
class _profiling_active:
    is_active=False
    
#check the env. var. 
def _is_profiling_enabled():
    return os.environ.get('ENABLE_PROFILING') == 'true'

#get the output directory
def _get_output():
    env_val = os.environ.get('PROFILE_OUTPUT_DIR')
    if env_val: 
        return env_val
    return PROFILE_OUTPUT_DIR

#get the sampling interval of pyinstrument (larger interval=lower overhead)
def _get_pyinstrument_interval():
    env_val = os.environ.get('PROFILING_INTERVAL')
    if env_val: 
        return float(env_val)
    return PROFILER_INTERVAL


def enable_profiling():
    os.environ['ENABLE_PROFILING'] = 'true'
    print("Profiling: Enabled.")

def disable_profiling():
    if 'ENABLE_PROFILING' in os.environ:
        del os.environ['ENABLE_PROFILING']
    print("Profiling: Disabled.")

def set_output(outdir):
    os.environ['PROFILE_OUTPUT_DIR'] = str(outdir)

def set_interval(interval):
    os.environ['PROFILING_INTERVAL'] = str(interval)

def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _is_profiling_enabled() and not _profiling_active.is_active:
            _profiling_active.is_active=True
            print(f"Profiling: Start profiling for {func.__module__}.{func.__name__}")
            interval=_get_pyinstrument_interval()
            print(f'Profiling: Sampling Interval {interval} s ')
            outdir=_get_output()
            os.makedirs(outdir, exist_ok=True)
            profiler = Profiler(interval=interval)
            profiler.start()
            #try:
            result = func(*args, **kwargs)
            #finally:
            profiler.stop()
            _profiling_active.is_active=False
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{func.__module__.replace('.', '_')}_{func.__name__}_{timestamp}.html"
            output_path = os.path.join(outdir, filename)
            profiler.write_html(output_path)
            print(f"Profiling: Data saved in {output_path}")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

