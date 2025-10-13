"""Database connection management with connection pooling."""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agent.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connection pool."""
    
    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
    
    def initialize(self, min_size: int = 2, max_size: int = 10):
        """Initialize connection pool."""
        if self._pool is not None:
            logger.warning("Connection pool already initialized")
            return
        
        logger.info(f"Initializing database connection pool (min={min_size}, max={max_size})")
        
        self._pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=min_size,
            max_size=max_size,
            timeout=30,
            kwargs={
                "row_factory": dict_row,  # Return rows as dictionaries
                "autocommit": False,
            }
        )
        
        logger.info("Database connection pool initialized successfully")
    
    def close(self):
        """Close connection pool."""
        if self._pool:
            logger.info("Closing database connection pool")
            self._pool.close()
            self._pool = None
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """Get a connection from the pool (context manager)."""
        if self._pool is None:
            self.initialize()
        
        with self._pool.connection() as conn:
            yield conn
    
    @contextmanager
    def get_cursor(self):
        """Get a cursor from the pool (context manager)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                yield cur


# Global database manager instance
db = DatabaseManager()


@contextmanager
def get_db_connection() -> Generator[Connection, None, None]:
    """Convenience function to get a database connection."""
    with db.get_connection() as conn:
        yield conn


@contextmanager
def get_db_cursor():
    """Convenience function to get a database cursor."""
    with db.get_cursor() as cur:
        yield cur
