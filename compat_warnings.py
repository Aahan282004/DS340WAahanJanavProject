"""
Shared warning suppression utilities for the FinBERT-LSTM project.

urllib3 v2 raises NotOpenSSLWarning whenever LibreSSL backs Python's ssl
module (macOS default). This helper silences that noisy warning so scripts
can run without cluttering the console, while leaving all other warnings
untouched.
"""

from __future__ import annotations

import warnings

try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover - fallback for very old urllib3 versions
    NotOpenSSLWarning = None  # type: ignore


if NotOpenSSLWarning is not None:
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
