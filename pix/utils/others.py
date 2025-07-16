import sys
import time
from functools import wraps
import os
import datetime
from socket import gethostname
from operator import eq
import pickle

def list_cat(ll):
    '''
    joint a iterable of lists into a single list.
    '''
    return sum(ll, [])

def save_load(filename, obj=None, is_save=True):
    folder = os.path.split(filename)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    if is_save:
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    else:
        with open(filename, 'rb') as file:
            return pickle.load(file)

def switch(value, comp=eq):
    return lambda match: comp(match, value)

def time_to_str(dt):
    if dt < 60:
        t = "{:.4f} sec".format(dt)
    elif dt < 3600:
        t = "{:.4f} min".format(dt/60)
    else:
        t = "{:.4f} hour".format(dt/3600)
    return t

def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        dt = te - ts
        t = time_to_str(dt)

        print("%r took %s " % (f.__name__, t))
        sys.stdout.flush()
        return result

    return wrapper

def timing_with_return(f):
    """Decorator for measuring the execution time of methods, added to fun return."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        dt = te - ts
        t = time_to_str(dt)

        print("%r took %s " % (f.__name__, t))
        sys.stdout.flush()
        return result, dt

    return wrapper

    
def redirect_log_file(log_root = "./log", exp_name="exp0"):
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    t = str(datetime.datetime.now())
    
    out_file = os.path.join(log_root, t[2:][:-7] + "  " + exp_name + ".txt")
    out_file = out_file.replace(":", "-")
    print("Redirect log to: ", out_file, flush=True)
    sys.stdout = open(out_file, 'a')
    sys.stderr = open(out_file, 'a')
    print("Start time:", t)
    print("Running at:", gethostname(), "pid=", os.getpid(), flush=True)
    return out_file

