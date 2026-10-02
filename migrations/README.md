# SQLite migrations

Numbered SQL migrations will live in this directory beginning with M2.

Naming convention:

~~~text
001_initial.sql
002_<description>.sql
003_<description>.sql
~~~

Schema changes must be forward migrations rather than ad-hoc `CREATE TABLE IF NOT EXISTS` changes.
