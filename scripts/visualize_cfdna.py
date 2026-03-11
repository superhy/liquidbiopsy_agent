#!/usr/bin/env python
from __future__ import annotations

import sys

from run_cfdna_plot_suite import main


if __name__ == "__main__":
    print(
        "[DEPRECATED] Use 'python scripts/run_cfdna_plot_suite.py' instead of "
        "'python scripts/visualize_cfdna.py'.",
        file=sys.stderr,
    )
    main()
