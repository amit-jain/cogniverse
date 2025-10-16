"""
Retry utilities with exponential backoff and jitter.

Provides production-ready retry mechanisms for handling transient failures.
"""

import logging
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, Tuple, Type, TypeVar, Union

from cogniverse_core.common.utils.async_polling import wait_for_operation_complete

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    exceptions: Tuple[Type[Exception], ...] = (Exception,)

    def should_retry(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return isinstance(exception, self.exceptions)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = min(
            self.initial_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay,
        )

        if self.jitter:
            # Add jitter to prevent thundering herd
            delay = delay * (0.5 + random.random() * 0.5)

        return delay


def retry_with_backoff(
    func: Optional[Callable[..., T]] = None,
    *,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
    """
    Decorator for retrying functions with exponential backoff

    Args:
        func: Function to retry
        config: Retry configuration
        on_retry: Callback called on each retry (exception, attempt)
        on_failure: Callback called on final failure (exception)

    Returns:
        Decorated function or decorator

    Example:
        @retry_with_backoff(config=RetryConfig(max_attempts=5))
        def fetch_data():
            return requests.get(url)

        # Or with callbacks
        @retry_with_backoff(
            on_retry=lambda e, a: logger.warning(f"Retry {a}: {e}"),
            on_failure=lambda e: logger.error(f"Failed: {e}")
        )
        def process_item(item):
            return api.process(item)
    """
    if config is None:
        config = RetryConfig()

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return f(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e) or attempt == config.max_attempts:
                        if on_failure:
                            on_failure(e)
                        raise

                    delay = config.get_delay(attempt)

                    if on_retry:
                        on_retry(e, attempt)
                    else:
                        logger.warning(
                            f"Retry {attempt}/{config.max_attempts} for {f.__name__} "
                            f"after {type(e).__name__}: {e}. Waiting {delay:.2f}s"
                        )

                    wait_for_operation_complete(delay, f"retry backoff for {f.__name__}")

            # Should never reach here, but for safety
            if last_exception:
                raise last_exception

        return wrapper

    # Handle being called with or without arguments
    if func is None:
        return decorator
    else:
        return decorator(func)


class RetryableOperation:
    """
    Context manager for retryable operations

    Example:
        with RetryableOperation(config=RetryConfig(max_attempts=3)) as retry:
            result = retry.execute(lambda: api.call())
    """

    def __init__(
        self, config: Optional[RetryConfig] = None, correlation_id: Optional[str] = None
    ):
        self.config = config or RetryConfig()
        self.correlation_id = correlation_id or "default"
        self._attempt = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, operation: Callable[[], T]) -> T:
        """Execute operation with retry logic"""
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            self._attempt = attempt

            try:
                return operation()

            except Exception as e:
                last_exception = e

                if (
                    not self.config.should_retry(e)
                    or attempt == self.config.max_attempts
                ):
                    logger.error(
                        f"[{self.correlation_id}] Operation failed after "
                        f"{attempt} attempts: {e}"
                    )
                    raise

                delay = self.config.get_delay(attempt)
                logger.warning(
                    f"[{self.correlation_id}] Retry {attempt}/{self.config.max_attempts} "
                    f"after {type(e).__name__}. Waiting {delay:.2f}s"
                )

                wait_for_operation_complete(delay, f"retry backoff for operation {self.correlation_id}")

        if last_exception:
            raise last_exception


def create_retry_decorator(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable:
    """
    Create a retry decorator with specific configuration

    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Tuple of exceptions to retry on
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Retry decorator

    Example:
        http_retry = create_retry_decorator(
            max_attempts=5,
            exceptions=(requests.RequestException,),
            initial_delay=0.5
        )

        @http_retry
        def fetch_data(url):
            return requests.get(url)
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        exceptions=exceptions,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )

    return lambda func: retry_with_backoff(func, config=config)
