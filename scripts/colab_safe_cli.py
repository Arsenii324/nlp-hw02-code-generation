#!/Users/a2mogus/.local/share/uv/tools/google-colab-cli/bin/python
"""Installed Colab CLI with Python debug logging disabled before imports."""
import logging
logging.disable(logging.CRITICAL)
from colab_cli.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
