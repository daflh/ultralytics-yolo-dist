#!/usr/bin/env python3

"""
YOLO command line tool replacement. Use it just like the original 'yolo' command.
Example usage:
    $ yolo.py task=detect mode=train model=yolo11n.pt data=coco.yaml epochs=100
"""

from ultralytics.cfg import entrypoint
import sys

args = sys.argv[1:]

entrypoint(" ".join(args))
