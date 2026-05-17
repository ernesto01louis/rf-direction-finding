# Guacamole PostgreSQL schema

Guacamole's database schema is **generated**, not committed — it is versioned
with the `guacamole/guacamole` image and shipping a stale copy invites drift.

The `guacamole` Ansible role generates it before first start:

```sh
docker run --rm guacamole/guacamole:1.5.5 \
    /opt/guacamole/bin/initdb.sh --postgresql > postgres/initdb/001-schema.sql
```

`postgres/initdb/` is mounted into the PostgreSQL container's
`/docker-entrypoint-initdb.d`, so the schema is applied automatically on the
first (empty-volume) start. Both `postgres/initdb/` and `postgres/data/` are
git-ignored.

For a standalone deployment, run the command above by hand before
`docker compose up -d`.
