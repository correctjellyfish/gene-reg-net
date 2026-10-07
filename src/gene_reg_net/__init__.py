"""
A Library for working with gene regulatory networks
"""
from importlib.metadata import version

__author__ = "Braden Griebel"
__version__ = version("metworkpy")

__all__ = ["GRN"]

from .network_class import GRN
