#!/usr/bin/env python3
"""
Database initialization script for Smart Doc Agent.

This script connects to PostgreSQL and runs all SQL migration files
in the sql/ directory to set up the schema.

Usage:
    python scripts/init_db.py
"""

import os
import sys
from pathlib import Path
import psycopg
from psycopg import sql

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_db_connection_string() -> str:
    """Build database connection string from environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "doc")
    password = os.getenv("DB_PASSWORD", "doc")
    database = os.getenv("DB_NAME", "docdb")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def run_sql_file(conn: psycopg.Connection, sql_file: Path) -> None:
    """Execute a SQL file."""
    print(f"  Running {sql_file.name}...")
    
    with open(sql_file, 'r') as f:
        sql_content = f.read()
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql_content)
        conn.commit()
        print(f"  ✓ {sql_file.name} completed")
    except Exception as e:
        conn.rollback()
        print(f"  ✗ Error in {sql_file.name}: {e}")
        raise


def init_database():
    """Initialize the database schema."""
    print("=" * 60)
    print("Smart Doc Agent - Database Initialization")
    print("=" * 60)
    
    # Get SQL files directory
    sql_dir = project_root / "sql"
    if not sql_dir.exists():
        print(f"Error: SQL directory not found at {sql_dir}")
        sys.exit(1)
    
    # Get all SQL files in order
    sql_files = sorted(sql_dir.glob("*.sql"))
    if not sql_files:
        print(f"Error: No SQL files found in {sql_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(sql_files)} SQL migration files:")
    for f in sql_files:
        print(f"  - {f.name}")
    
    # Connect to database
    conn_string = get_db_connection_string()
    print(f"\nConnecting to database...")
    print(f"  Host: {os.getenv('DB_HOST', 'localhost')}")
    print(f"  Port: {os.getenv('DB_PORT', '5432')}")
    print(f"  Database: {os.getenv('DB_NAME', 'docdb')}")
    print(f"  User: {os.getenv('DB_USER', 'doc')}")
    
    try:
        with psycopg.connect(conn_string) as conn:
            print("✓ Connected successfully\n")
            
            # Run each SQL file in order
            print("Running migrations:")
            for sql_file in sql_files:
                run_sql_file(conn, sql_file)
            
            # Verify installation
            print("\nVerifying installation:")
            with conn.cursor() as cur:
                # Check pgvector
                cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
                result = cur.fetchone()
                if result:
                    print(f"  ✓ pgvector extension: v{result[1]}")
                else:
                    print("  ✗ pgvector extension not found")
                
                # Check tables
                cur.execute("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public' 
                    ORDER BY tablename
                """)
                tables = [row[0] for row in cur.fetchall()]
                print(f"  ✓ Created {len(tables)} tables:")
                for table in tables:
                    print(f"    - {table}")
                
                # Check views
                cur.execute("""
                    SELECT viewname FROM pg_views 
                    WHERE schemaname = 'public'
                    ORDER BY viewname
                """)
                views = [row[0] for row in cur.fetchall()]
                if views:
                    print(f"  ✓ Created {len(views)} views:")
                    for view in views:
                        print(f"    - {view}")
            
            print("\n" + "=" * 60)
            print("✓ Database initialization completed successfully!")
            print("=" * 60)
            
    except psycopg.OperationalError as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if PostgreSQL container is running: docker compose ps db")
        print("  2. Verify environment variables are set correctly")
        print("  3. Ensure database 'docdb' exists")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    init_database()
