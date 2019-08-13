#!/usr/bin/env python3

from absl.testing import absltest

import numpy as np
from additive_convolution import batch_add_conv_1d

class UnitTests(absltest.TestCase):

    def test_simple_case(self):
        A = np.array([
            [0,0,7,5,0,4,8,3,0,0],
            [1,4,8,6,5,4,0,3,2,0]
        ], np.float32)
        b = np.array([
            -3,-1,0,-1,-2
        ], np.float32)
        C_expected = np.array([
            [5,6,7,6,6,7,8,7,5,0],
            [6,7,8,7,5,4,3,3,2,1]
        ], np.float32)

        C = batch_add_conv_1d(
            np.array([A]), b
        )[0]

        self.assertTrue(np.allclose(C_expected, C))


if __name__ == '__main__':
    absltest.main()
