"""Database lifecycle management for the evaluator."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import psycopg
from psycopg import sql

from agent.config import settings
from agent.db import db

from agent.evaluator.models import ScenarioResult

if TYPE_CHECKING:  # pragma: no cover - type check helper
    from agent.evaluator.harness import EvaluatorConfig


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Create, migrate, and clean up ephemeral evaluation databases."""

    def __init__(self, config: "EvaluatorConfig", original_db_name: str) -> None:
        self._config = config
        self._original_db_name = original_db_name
        self._ephemeral_db_name: Optional[str] = None

    def prepare(self) -> ScenarioResult:
        start = time.perf_counter()
        db_name = self._build_test_db_name()
        details: list[str] = []
        metrics = {"database": db_name}
        status = "passed"
        try:
            self._create_fresh_database(db_name)
            self._apply_schema_to_database(db_name)
            self._switch_database(db_name)
            self._ephemeral_db_name = db_name
            details.append(f"created:{db_name}")
        except FileNotFoundError:
            logger.warning("Schema scripts not found; reusing existing database %s", self._original_db_name)
            details.append("schema_dir_missing:using_existing_db")
            metrics["database"] = self._original_db_name
            self._cleanup_database(db_name, details)
            self._ephemeral_db_name = None
            self._restore_original_database()
        except Exception as exc:  # noqa: BLE001 - setup must not crash evaluator
            logger.exception("Failed to prepare test database %s", db_name)
            status = "failed"
            details.append(f"error:{exc}")
            self._cleanup_database(db_name, details)
            self._restore_original_database()
        duration = time.perf_counter() - start
        return ScenarioResult(
            name="db_setup",
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=duration,
        )

    def teardown(self) -> Optional[ScenarioResult]:
        if not self._ephemeral_db_name:
            return None

        start = time.perf_counter()
        details: list[str] = []
        status = "passed"
        db_name = self._ephemeral_db_name

        try:
            db.close()
        except Exception as exc:  # pragma: no cover - defensive close
            status = "warn"
            details.append(f"pool_close:{exc}")

        try:
            self._drop_database(db_name)
            details.append(f"dropped:{db_name}")
        except Exception as exc:  # pragma: no cover - best effort cleanup
            status = "failed"
            details.append(f"drop_error:{exc}")

        self._restore_original_database()
        self._ephemeral_db_name = None
        duration = time.perf_counter() - start

        return ScenarioResult(
            name="db_cleanup",
            status=status,
            metrics={"database": db_name},
            details=details,
            duration_seconds=duration,
        )

    def current_db(self) -> Optional[str]:
        return self._ephemeral_db_name

    def _cleanup_database(self, db_name: str, details: list[str]) -> None:
        try:
            self._drop_database(db_name)
        except Exception as cleanup_exc:  # pragma: no cover - best effort cleanup
            details.append(f"drop_cleanup_error:{cleanup_exc}")

    def _build_test_db_name(self) -> str:
        base = f"{self._original_db_name}_eval_{self._config.run_id}".lower()
        sanitized = re.sub(r"[^a-z0-9_]+", "_", base)
        if len(sanitized) > 63:
            sanitized = sanitized[:63]
        return sanitized.rstrip("_") or f"{self._original_db_name}_eval"

    def _restore_original_database(self) -> None:
        db.close()
        object.__setattr__(settings, "db_name", self._original_db_name)
        os.environ["DB_NAME"] = self._original_db_name
        os.environ["DATABASE_URL"] = settings.database_url

    def _switch_database(self, db_name: str) -> None:
        db.close()
        object.__setattr__(settings, "db_name", db_name)
        os.environ["DB_NAME"] = db_name
        os.environ["DATABASE_URL"] = settings.database_url

    def _create_fresh_database(self, db_name: str) -> None:
        conninfo = self._build_conninfo(self._original_db_name)
        with psycopg.connect(conninfo, autocommit=True) as conn:
            drop_sql = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(db_name))
            create_sql = sql.SQL(
                "CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'"
            ).format(sql.Identifier(db_name), sql.Identifier(settings.db_user))
            conn.execute(drop_sql)
            conn.execute(create_sql)

    def _drop_database(self, db_name: str) -> None:
        conninfo = self._build_conninfo(self._original_db_name)
        with psycopg.connect(conninfo, autocommit=True) as conn:
            drop_sql = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(db_name))
            conn.execute(drop_sql)

    def _apply_schema_to_database(self, db_name: str) -> None:
        configured_dir = os.getenv("EVAL_SCHEMA_DIR")
        candidate_dirs = []
        if configured_dir:
            candidate_dirs.append(Path(configured_dir))
        candidate_dirs.extend(
            [
                Path("/app/sql"),
                Path(__file__).resolve().parents[2] / "sql",
                Path(__file__).resolve().parents[3] / "sql",
                Path.cwd() / "sql",
            ]
        )

        schema_dir: Optional[Path] = None
        for candidate in candidate_dirs:
            if candidate.exists():
                schema_dir = candidate
                break

        if not schema_dir:
            raise FileNotFoundError("No schema scripts directory found")

        scripts = sorted(schema_dir.glob("*.sql"))
        if not scripts:
            raise FileNotFoundError(f"No schema scripts found in {schema_dir}")

        conninfo = self._build_conninfo(db_name)
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                for script_path in scripts:
                    sql_text = script_path.read_text(encoding="utf-8").strip()
                    if not sql_text:
                        continue
                    cur.execute(sql_text)
            conn.commit()

    def _build_conninfo(self, database: str) -> str:
        return (
            f"postgresql://{settings.db_user}:{settings.db_password}"
            f"@{settings.db_host}:{settings.db_port}/{database}"
        )
