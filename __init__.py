import logging
import sys

if sys.version > "3":
    PY2 = False
else:
    PY2 = True

__version__ = '0.0.1'
__all__ = ['constants', 'atomic_mass', 'atom', 'config']

# Initialize logger.
logger = logging.getLogger("config_tools")
logger.setLevel(logging.INFO)
console_hdlr = logging.StreamHandler()
console_hdlr.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s %(levelname)-8s %(message)s")
console_hdlr.setFormatter(formatter)
logger.addHandler(console_hdlr)
