#!/usr/bin/env python3
from ultralytics.cfg import entrypoint
import sys

args = sys.argv[1:]

entrypoint(" ".join(args))
