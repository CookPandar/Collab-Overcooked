import random
import time

import openai


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                print(e)
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper
