# DIP SIGNAL — Brand System

## 1. Brand concept

**DIP SIGNAL** — data-first, evidence-led investment analysis. The name encodes the core question: is there a dip, and what does history say about it?

## 2. Positioning

"Market pressure, measured."

No predictions. No promises. Historical evidence, clearly presented.

## 3. Verbal personality

- **Decisive** — direct findings, no hedging beyond what the data requires
- **Evidence-led** — every claim is sourced from historical data
- **Energetic** — compressed, uppercase section heads; tight copy
- **Honest** — explicit disclaimers, no financial advice framing

## 4. Palette — dark surfaces + iridescent accents

| Token | Hex | Role |
|---|---|---|
| `BG_CANVAS` | `#07070A` | Page background |
| `BG_SURFACE` | `#111218` | Card / container background |
| `BG_SURFACE_RAISED` | `#1A1C24` | Elevated surface, tooltips |
| `BG_SURFACE_ACTIVE` | `#232631` | Active tab, selected state |
| `BORDER` | `#343746` | Default border |
| `BORDER_STRONG` | `#505465` | Emphasized divider |
| `TEXT_PRIMARY` | `#F7F7F2` | Headings, primary labels |
| `TEXT_SECONDARY` | `#C3C6D1` | Captions, metadata |
| `TEXT_MUTED` | `#959AAA` | Placeholder, disabled labels |
| `ACCENT_CYAN` | `#42E8FF` | Primary accent — dip signal, CTAs |
| `ACCENT_MAGENTA` | `#FF4DC4` | Tiered strategy |
| `ACCENT_LIME` | `#D8FF4F` | Tertiary accent |
| `ACCENT_ORANGE` | `#FF914D` | Invest-now / FX contribution |
| `POSITIVE` | `#37D39A` | Gains, beats baseline |
| `NEGATIVE` | `#FF5F73` | Losses, drawdowns |
| `WARNING` | `#FFB84A` | Approaching threshold, amber state |
| `NEUTRAL` | `#A7ACB9` | DCA baseline, idle state |

Brand gradient (decoration only): `linear-gradient(115deg, #42E8FF 0%, #6576FF 30%, #FF4DC4 65%, #D8FF4F 100%)`

## 5. Typography

- **Display / headings**: Arial Narrow, Roboto Condensed, or Arial (condensed sans-serif)
- **Body**: Inter, Segoe UI, system sans-serif
- **Numeric**: `font-variant-numeric: tabular-nums` on all metric values

## 6. Components

| Component | File | Purpose |
|---|---|---|
| `page_header()` | `ui/components.py` | Gradient accent bar + uppercase heading |
| `signal_card()` | `ui/components.py` | Hero drawdown card with 2×2 metric grid |
| `conclusion_banner()` | `ui/components.py` | Full-width result strip with colored left border |
| `strategy_comparison_cards()` | `ui/components.py` | Side-by-side DCA vs dip cards |
| `dip_badge()` | `ui/components.py` | Inline status badge with border color |
| `brand_motif()` | `ui/components.py` | SVG signal-line motif for decoration |
| `sample_size_badge()` | `ui/components.py` | Evidence-quality badge |

## 7. Chart encoding

| Strategy | Color | Dash |
|---|---|---|
| DCA / Invest monthly | `NEUTRAL (#A7ACB9)` | solid |
| Wait for dip | `ACCENT_CYAN (#42E8FF)` | dash |
| Tiered deployment | `ACCENT_MAGENTA (#FF4DC4)` | dashdot |
| Invest now | `ACCENT_ORANGE (#FF914D)` | dot |

Chart backgrounds: transparent paper, `BG_SURFACE` plot area. Grid lines: `BORDER`. Axis text: `TEXT_SECONDARY`.

## 8. Prohibited

- White or near-white surfaces (`#FFFFFF`, `#F9FAFB`, `#EFF6FF`, `#F3F4F6`)
- Dark text on dark background (`#111827` in CSS)
- Mascot names (Monthly Machine, Cash Goblin, Dip Buffet, Market Arcade)
- "Class of One" (brand IP)
- FutureBrand assets, World Athletics references
- Casino or gambling language
- Future-performance promises or investment advice
