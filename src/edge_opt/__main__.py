"""Allow ``python -m edge_opt ...`` as an alias for the ``edge-opt`` command."""

from edge_opt.cli import main

raise SystemExit(main())
