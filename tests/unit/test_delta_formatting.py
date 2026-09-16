"""Delta presentation must not round real differences away or mis-word ties.

Two audited display defects:

- ``f"{delta:+,.0f}"`` renders any difference below EUR 1 as "0";
- ``dca_wins = dca_wealth >= dip_wealth`` reports an exact tie as the baseline
  winning "by EUR 0".
"""

from __future__ import annotations

import pytest

from ui.formatting import compare_outcomes, fmt_delta


class TestFmtDelta:
    def test_small_nonzero_delta_is_not_rendered_as_zero(self):
        out = fmt_delta(0.43, 7_000.0)

        assert "0.43" in out
        assert out.strip() != "EUR +0"

    def test_delta_shows_both_absolute_and_relative(self):
        out = fmt_delta(0.43, 7_000.0)

        assert "EUR" in out
        assert "%" in out

    def test_sign_is_explicit_for_gains_and_losses(self):
        assert "+" in fmt_delta(10.0, 1_000.0)
        assert "-" in fmt_delta(-10.0, 1_000.0)

    def test_thousands_separator_is_applied(self):
        assert "1,234.56" in fmt_delta(1234.56, 100_000.0)

    def test_zero_baseline_does_not_raise(self):
        out = fmt_delta(5.0, 0.0)

        assert "5.00" in out


class TestCompareOutcomes:
    def test_exact_tie_is_not_described_as_dca_winning(self):
        verdict, diff = compare_outcomes(1_000.0, 1_000.0, "Dip", "DCA")

        assert verdict == "Effectively equal"
        assert diff == 0.0

    def test_negligible_difference_is_a_tie(self):
        verdict, _ = compare_outcomes(1_000.000_000_1, 1_000.0, "Dip", "DCA")

        assert verdict == "Effectively equal"

    def test_candidate_ahead_is_named(self):
        verdict, diff = compare_outcomes(1_100.0, 1_000.0, "Dip", "DCA")

        assert verdict == "Dip ahead"
        assert diff == 100.0

    def test_baseline_ahead_is_named(self):
        verdict, diff = compare_outcomes(900.0, 1_000.0, "Dip", "DCA")

        assert verdict == "DCA ahead"
        assert diff == -100.0

    def test_tolerance_is_relative_not_a_flat_euro_amount(self):
        """EUR 5 is noise on a EUR 10m portfolio but decisive on EUR 100."""
        big, _ = compare_outcomes(10_000_005.0, 10_000_000.0, "Dip", "DCA")
        small, _ = compare_outcomes(105.0, 100.0, "Dip", "DCA")

        assert big == "Effectively equal"
        assert small == "Dip ahead"

    def test_a_real_sub_euro_difference_is_still_a_win_at_small_scale(self):
        verdict, diff = compare_outcomes(100.43, 100.0, "Dip", "DCA")

        assert verdict == "Dip ahead"
        assert diff == pytest.approx(0.43)
