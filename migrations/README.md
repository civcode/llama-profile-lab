# SQLite migrations

Numbered SQL migrations live in this directory.

Naming convention:

~~~text
001_initial.sql
002_<description>.sql
003_<description>.sql
~~~

The migration runner records each applied migration's version, name, and SHA-256 checksum in the `schema_migration` table. Applied migration files are immutable: changing the contents of an already-applied migration is treated as an error.

Schema changes must therefore be forward migrations rather than ad-hoc `CREATE TABLE IF NOT EXISTS` changes.


## Migration policy

The V1 schema is migrations 001 through 005. V2-M1 adds migration 006 for multi-model deployment persistence and migration 007 for explainable deployment rejection history. The current schema version is therefore 7. The test suite creates databases at every historical schema prefix and verifies that each one upgrades through the current migration set with SQLite integrity and foreign-key checks passing.

Operational rules:

- never edit an applied migration;
- add a new sequential migration for every schema change;
- keep migrations deterministic and independent of local machine state;
- do not delete historical migration files from a release line;
- run `uv run --frozen pytest tests/test_db_migrations.py` before release;
- after upgrading a real database, run `llprof database check --database PATH`.

The migration runner stores the SHA-256 checksum of every applied migration. A checksum mismatch is treated as corruption/configuration drift and stops startup rather than silently accepting a different schema history.

Archives store a complete SQLite snapshot plus the schema version recorded at snapshot time. Restore verifies the archived database's `schema_migration` version against the manifest before writing the destination database. A restored older archive may subsequently be upgraded by the normal migration runner.
