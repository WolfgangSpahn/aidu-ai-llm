# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
# If you use this software in academic work, citation of the original author is requested.
# engine.py

from abc import ABC


class Engine(ABC):
    # has a static method called process that takes an input and returns an output
    process = staticmethod(lambda input, **kwargs: input)

    def evaluate(self, input, context, config=None):
        self.process(input, context=context, config=config)
