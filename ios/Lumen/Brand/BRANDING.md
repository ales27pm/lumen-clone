# Lumen visual identity

## Concept

Lumen is the light source. The assistant appears as a compact intelligent orb that illuminates a darker surrounding field. The brand should feel private, on-device, alive, and precise rather than decorative.

## Core visual rules

- Backgrounds stay near-black with deep blue-violet undertones.
- The assistant mark emits warm light from its core, shifting toward cool blue and violet at the edge.
- Light should originate from the assistant itself, not from external UI decoration.
- Use soft radial illumination, not heavy neon outlines.
- Surfaces are translucent dark glass with subtle borders.
- The wordmark is quiet, rounded, and technical.

## Palette

| Token | Hex | Purpose |
|---|---:|---|
| Midnight | `#03040A` | deepest background |
| Deep Space | `#06080D` | primary dark field |
| Ink | `#090B12` | elevated dark surfaces |
| Lumen | `#FFEAA3` | assistant light core |
| Ember | `#FFC75C` | warm source color |
| Corona | `#A3CCFF` | cool halo |
| Plasma | `#788FFF` | technical blue accent |
| Violet | `#A36EFF` | peripheral glow |

## Included assets

- `lumen-assistant-mark.png` — square app-mark source.
- `lumen-vertical-logo.png` — vertical logo lockup source.
- `lumen-wordmark-lockup.png` — horizontal logo lockup source.
- `LumenBrand.swift` — native SwiftUI brand components.

## SwiftUI components

- `LumenAssistantMark`
- `LumenBrandBackground`
- `LumenLightBeam`

These are preferred in-app because they scale cleanly and preserve dynamic illumination without raster assets.
