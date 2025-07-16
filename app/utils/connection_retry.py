"""
Connection retry utility for handling Supabase connection errors.
"""

import time
import functools
from typing import Any, Callable
import httpx
from httpcore import RemoteProtocolError

def retry_on_connection_error(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator to retry functions on connection errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor (delay = backoff_factor * (2 ** attempt))
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (httpx.RemoteProtocolError, RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = backoff_factor * (2 ** attempt)
                        print(f"Connection error on attempt {attempt + 1}: {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        print(f"Connection failed after {max_retries + 1} attempts: {e}")
                        break
                except Exception as e:
                    # For non-connection errors, don't retry
                    print(f"Non-connection error: {e}")
                    raise
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator

def safe_supabase_call(func: Callable, *args, **kwargs) -> Any:
    """
    Safely call a Supabase function with retry logic.
    
    Args:
        func: Function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call or None if all retries failed
    """
    try:
        @retry_on_connection_error(max_retries=3, backoff_factor=0.5)
        def _safe_call():
            return func(*args, **kwargs)
        
        return _safe_call()
    except Exception as e:
        print(f"Supabase call failed permanently: {e}")
        return None