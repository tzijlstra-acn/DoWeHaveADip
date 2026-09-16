"""Redirect module — raises on import so stale page files fail clearly.

The simulation logic has moved:

- ``conditional_path_bootstrap`` → removed (used ETF as signal; unpaired comparisons)
- ``run_parameter_sweep``         → removed (used run_wait_for_dip internally)
- Episode-level analysis          → ``dipdca.quant.ath_episodes``
- Paired bootstrap                → ``dipdca.quant.episode_bootstrap``

The original implementation is preserved at ``dipdca.quant.legacy_monte_carlo``
for reference only.
"""

raise ImportError(
    "dipdca.quant.monte_carlo has been quarantined. "
    "Use dipdca.quant.ath_episodes for the event study and "
    "dipdca.quant.episode_bootstrap for the paired ATH-episode bootstrap. "
    "If you genuinely need the old implementation for a one-off comparison, "
    "import dipdca.quant.legacy_monte_carlo directly."
)
