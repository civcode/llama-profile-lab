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
