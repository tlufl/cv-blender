import logging
import functools
from typing import Callable, TypeVar, ParamSpec, Any

logging.basicConfig(
    filename="cv.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def logg(
    log_fn: Callable[[Any], str] = repr,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def logg_it(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)

            if result is not None:
                logger.info(f"Function '{func.__name__}' returned: {log_fn(result)}")

            return result

        return wrapper

    return logg_it
