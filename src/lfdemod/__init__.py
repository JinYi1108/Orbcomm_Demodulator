"""Public package for the LFdemod project.

The original :mod:`orbdemod` package remains available for backward
compatibility.  New user-facing code should import from :mod:`lfdemod`.
"""

from orbdemod import __version__

__all__ = ["__version__"]
