import unittest

from api.backtesting import (
    calc_directional_accuracy,
    calc_mape,
    safe_pct_change,
    summarize_leverage_outcomes,
    summarize_signal_outcomes,
)


class BacktestingTests(unittest.TestCase):
    def test_safe_pct_change(self):
        self.assertAlmostEqual(safe_pct_change(100, 110), 10.0)
        self.assertEqual(safe_pct_change(0, 110), 0.0)

    def test_directional_accuracy(self):
        result = calc_directional_accuracy([110, 90], [120, 80], [100, 100])
        self.assertEqual(result, 1.0)

    def test_mape(self):
        self.assertAlmostEqual(calc_mape([100, 200], [110, 190]), 7.1770, places=3)

    def test_signal_summary(self):
        summary = summarize_signal_outcomes([
            {'future_return_pct': 5.0},
            {'future_return_pct': -2.0},
        ])
        self.assertEqual(summary['samples'], 2)
        self.assertEqual(summary['hit_rate'], 0.5)

    def test_leverage_summary(self):
        summary = summarize_leverage_outcomes([
            {'realized_abs_return_pct': 3.0, 'stop_loss_pct': 2.0, 'recommended_leverage': 2},
            {'realized_abs_return_pct': 1.0, 'stop_loss_pct': 2.0, 'recommended_leverage': 4},
        ])
        self.assertEqual(summary['samples'], 2)
        self.assertEqual(summary['stop_breach_rate'], 0.5)
        self.assertEqual(summary['avg_recommended_leverage'], 3.0)


if __name__ == '__main__':
    unittest.main()
