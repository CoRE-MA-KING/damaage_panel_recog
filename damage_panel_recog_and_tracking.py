#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point kept for backward compatibility.

This script delegates to the refactored package implementation so you can
swap tracking backends (motpy today, your own SORT-based tracker later)
without touching the detection / camera / UI code.
"""

from damage_panel_tracking.cli import main

if __name__ == "__main__":
    # パッケージ側のCLIエントリポイントへ起動を委譲する。
    raise SystemExit(main())
