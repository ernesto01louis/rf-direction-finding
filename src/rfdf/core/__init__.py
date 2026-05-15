"""Cross-cutting concerns for rfdf.

Modules in ``rfdf.core`` are policy + plumbing the algorithm and backend layers
share — distinct from :mod:`rfdf.hal`, which defines contracts only. Examples:
EIRP enforcement, logging configuration (Stage 3), error taxonomy.

Nothing in ``rfdf.core`` may import a backend module. Backends MAY import core.
"""
