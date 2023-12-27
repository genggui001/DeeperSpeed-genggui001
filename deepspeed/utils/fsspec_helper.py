import os
import time
import traceback

from upath import UPath
from wrapt_timeout_decorator import timeout
from functools import wraps


def timeout_and_retry(
    num_retries, 
    one_retry_timeout,
    finally_skip_error=False,
):
    def decorate(func):
        @wraps(func)

        def wrapper(*args, **kwargs):
            for i in range(num_retries):
                try:
                    # return timeout(one_retry_timeout)(func)(*args, **kwargs)
                    return func(*args, **kwargs)
                except:
                    from deepspeed.utils import logger
                    logger.info(f"retry on {func.__name__} {i+1}/{num_retries}.\n\n" + traceback.format_exc())

                    if i < num_retries - 1:
                        # retry
                        time.sleep(2 ** i)
                    else:
                        if finally_skip_error == False:
                            raise ValueError(f"{func.__name__} failed.\n\n" + traceback.format_exc())
            
        return wrapper
    return decorate

@timeout_and_retry(
    num_retries=int(os.environ.get("MFSSPEC_NUM_RETRIES", "5")),
    one_retry_timeout=int(os.environ.get("MFSSPEC_TIMEOUT", "1800")),
    finally_skip_error=False,
)
def glob_use_fsspec(
    base_path,
    pathname,
):
    return [str(pitem) for pitem in UPath(base_path).glob(pathname)]

@timeout_and_retry(
    num_retries=int(os.environ.get("MFSSPEC_NUM_RETRIES", "5")),
    one_retry_timeout=int(os.environ.get("MFSSPEC_TIMEOUT", "1800")),
    finally_skip_error=False,
)
def exists_use_fsspec(
    base_path,
):
    return UPath(base_path).exists()

