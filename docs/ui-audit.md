# UI Audit — Dark Theme Remediation

| Component | File | Problem | Status |
|---|---|---|---|
| Page background | `.streamlit/config.toml` + `ui/theme.py` | Dark Streamlit shell (`#0E1117`) + light CSS injection (`#FFFFFF` cards, `#111827` text) | Fixed — unified dark-first system |
| Metric cards | `ui/theme.py` | `#FFFFFF` card background on `#0E1117` shell | Fixed — `BG_SURFACE (#111218)` |
| Metric label text | `ui/theme.py` | `#6B7280` (readable on light, too dim on dark) | Fixed — `TEXT_MUTED` |
| Metric value text | `ui/theme.py` | `#111827` (dark text, invisible on dark bg) | Fixed — `TEXT_PRIMARY (#F7F7F2)` |
| Active tab | `ui/theme.py` | `#EFF6FF` light blue background | Fixed — `BG_SURFACE_ACTIVE` + `ACCENT_CYAN` border |
| Expander background | `ui/theme.py` | `#FFFFFF` background | Fixed — `BG_SURFACE` |
| Primary button | `ui/theme.py` | `#1A56DB` blue (OK contrast, but inconsistent) | Updated — `ACCENT_CYAN` with `BG_CANVAS` text |
| HR border | `ui/theme.py` | `#E5E7EB` (light gray, near-invisible on dark) | Fixed — `BORDER (#343746)` |
| Page header | `ui/components.py` | `#111827` heading, `#6B7280` subtitle, `#E5E7EB` border | Fixed — `TEXT_PRIMARY`, `TEXT_MUTED`, `BORDER` |
| Strategy cards | `ui/components.py` | Inline white/light card backgrounds | Fixed — `BG_SURFACE`, dark text tokens |
| Dip badge | `ui/components.py` | `#FEF2F2`/`#FFFBEB`/`#ECFDF5` light backgrounds | Fixed — `BG_SURFACE_RAISED` + colored border |
| Hero signal card | `app_pages/today.py` | `#FFFFFF` background, `#374151`/`#9CA3AF` text | Fixed — `BG_SURFACE`, `TEXT_SECONDARY`, `TEXT_MUTED` |
| Compare conclusion | `app_pages/compare.py` | `#F9FAFB` background, `#111827` text | Fixed — `conclusion_banner()` component |
| Chart background | `ui/charts.py` | `paper_bgcolor=BG_CARD (#FFFFFF)`, `plot_bgcolor=BG_PAGE` | Fixed — transparent paper + `BG_SURFACE` plot |
| Gauge chart | `ui/charts.py` | `bgcolor="#1E2130"`, `bordercolor="#3D4066"` hardcodes | Fixed — `BG_SURFACE`, `BORDER` tokens |
| Heatmap colorbar | `ui/charts.py` | `tickfont color="#FAFAFA"` | Updated — `TEXT_PRIMARY` token |
| Episode annotations | `ui/charts.py` | `color="#9CA3AF"` hardcode | Fixed — `TEXT_MUTED` token |
| Strategy colors | `ui/design_tokens.py` | Light-theme colors (blue primary, green positive) | Replaced — iridescent accent palette |
| Streamlit theme | `.streamlit/config.toml` | `primaryColor="#F47920"` (orange) mismatching new system | Fixed — `ACCENT_CYAN (#42E8FF)` |
