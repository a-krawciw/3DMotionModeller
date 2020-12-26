from unittest import TestCase
import numpy as np


class ArrayTestCase(TestCase):
    def assertArrayEqual(self, arr1, arr2):
        if not np.array_equal(arr1, arr2):
            self.fail(msg=f"{arr1} != {arr2}")

    def assertIsNan(self, val):
        if not np.isnan(val):
            self.fail(msg=f"{val} is not nan")

    def assertIsNotNan(self, val):
        if np.isnan(val):
            self.fail(msg=f"Expected not nan, got {val}")
