import unittest

from api.validators import (
    normalize_interval,
    normalize_limit,
    normalize_validation_horizon,
    normalize_validation_window,
    validate_interval,
    validate_ticker,
)


class ValidatorTests(unittest.TestCase):
    def test_validate_ticker(self):
        ok, value = validate_ticker('BTCUSDT')
        self.assertTrue(ok)
        self.assertEqual(value, 'BTCUSDT')

    def test_invalid_ticker(self):
        ok, value = validate_ticker('../../etc/passwd')
        self.assertFalse(ok)
        self.assertEqual(value, 'Invalid ticker format')

    def test_interval_normalization(self):
        self.assertEqual(normalize_interval('4h'), '4h')
        self.assertEqual(normalize_interval('bad'), '1d')
        self.assertTrue(validate_interval('1d')[0])
        self.assertFalse(validate_interval('bad')[0])

    def test_limit_clamping(self):
        self.assertEqual(normalize_limit('10'), 50)
        self.assertEqual(normalize_limit('9999'), 1000)

    def test_validation_window_horizon_clamping(self):
        self.assertEqual(normalize_validation_window('30'), 90)
        self.assertEqual(normalize_validation_horizon('999'), 30)


if __name__ == '__main__':
    unittest.main()
