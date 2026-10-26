"""Allow `python -m retail_warehouse` as a shortcut for the CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
