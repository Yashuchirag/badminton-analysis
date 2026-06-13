# Glass Arena UI Redesign

**Date:** 2026-06-12
**Scope:** `frontend/myApp/lib/theme.ts`, `app/index.tsx`, `app/live.tsx` (and a new `components/ShuttleBackground.tsx`)
**Status:** Approved by user, ready for implementation planning

---

## Overview

Replace the current flat indigo-light design with a "Glass Arena" dark theme: a deep court-green radial background with floating soft-realistic shuttlecock SVGs that repel from touch, frosted-glass cards for all content surfaces, Barlow Condensed scores, and monospace micro-labels. No new native modules are required beyond the packages already installed (`react-native-reanimated`, `react-native-svg`, `react-native-gesture-handler`).

---

## 1. Design Tokens (`lib/theme.ts`)

Replace all existing token values with the following. Every component reads from these tokens; no inline hex anywhere.

```
background:       #020a05          // near-black court green
backgroundGrad:   radial-gradient(circle at 70% 15%, #14532d 0%, #052e16 45%, #020a05 100%)
surface:          rgba(255,255,255,0.07)   // frosted glass tint
surfaceBorder:    rgba(255,255,255,0.18)
surfacePressed:   rgba(255,255,255,0.12)

textPrimary:      #ffffff
textMuted:        #86efac           // sage green
textMono:         #86efac           // same, signals monospace usage

accent:           #4ade80           // neon green (scores, live glow, progress bar)
accentDim:        rgba(74,222,128,0.18)
accentBorder:     rgba(74,222,128,0.45)

ctaBackground:    #16a34a           // dark green CTA button fill
ctaBorder:        #4ade80
ctaText:          #ffffff

destructive:      #dc2626           // stop button
warning:          #fbbf24           // calibrating banner
success:          #4ade80           // same as accent; reuse token

overlayDark:      rgba(2,10,5,0.72) // scrim behind modals
```

**Typography tokens:**
```
fontHeading:      Barlow Condensed  // scores, screen title
fontBody:         Barlow            // card body, buttons
fontMono:         Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' })
```

Monospace uses the platform system font (Menlo on iOS, monospace on Android). No new font package or asset needed.

---

## 2. Shuttle Background Component (`components/ShuttleBackground.tsx`)

A standalone component rendered behind all screen content via `StyleSheet.absoluteFillObject`.

**Visual:**
- Radial gradient background painted via a full-screen `react-native-svg` `<Svg>` using a `<RadialGradient>` def. `expo-linear-gradient` is not installed and not needed; `react-native-svg` 15.12.1 is already in `package.json` and supports radial gradients natively.
- Five shuttlecock SVGs at fixed positions, varying sizes (26dp, 30dp, 34dp, 38dp, 42dp), and varying base opacities (0.40, 0.50, 0.55, 0.65, 0.75). The SVG is the soft-realistic design: white feathers (`#f7f9f4`), `#cfd6cc` stroke, tan cork (`#e3cfa4`).

**Drift animation:**
- Each shuttle has an `Animated.ValueXY` (or `useSharedValue` pair) that runs a looping slow drift using `withRepeat(withSequence(withTiming(...), withTiming(...)))`.
- Each shuttle gets a unique phase offset (0ms, 900ms, 1800ms, 2700ms, 3600ms) and unique drift amplitude (±8dp to ±16dp X, ±10dp to ±20dp Y) so they move independently.
- Duration per half-cycle: 6000ms to 9000ms, easing `Easing.inOut(Easing.sine)`.

**Repel interaction:**
- A `PanGestureHandler` (or `GestureDetector` with `Gesture.Pan()` from `react-native-gesture-handler`) wraps the whole background.
- On `onGestureEvent`: for each shuttle compute distance from touch point. If distance < 120dp, add a repel offset: `repelOffset = (120 - distance) / 120 * 80 * unitVector`. Apply via an additional `useSharedValue` that overrides the drift position.
- On `onHandlerStateChange` (end/cancel): spring the repel offset back to `{x:0, y:0}` using `withSpring({ damping: 14, stiffness: 80 })`.
- The repel shared values are added to the drift animated values in the shuttle's `animatedStyle` transform.

**Performance:**
- All animations run on the UI thread via reanimated worklets. No JS-thread frame drops.
- Five shuttles maximum. The gesture handler covers the full screen but passes touches through to children via `simultaneousHandlers` so card taps still register.

---

## 3. Home Screen (`app/index.tsx`)

**Structure (top to bottom):**
1. `<ShuttleBackground />` — `position: absolute`, fills screen, z-index 0.
2. `<SafeAreaView>` wrapping the rest, z-index 1, transparent background.
3. Header row: app name "Game Tracker" in Barlow Condensed 22pt `#fff`, left-aligned, no background surface.
4. Spacer (flex: 1 or fixed 24dp).
5. **Action card** (frosted glass): two side-by-side tiles.
   - Left tile: "Analyze video" — upload icon (Ionicons `cloud-upload-outline`), Barlow 14pt label.
   - Right tile: "Go live" — icon `radio-outline`, Barlow 14pt label in `#4ade80`, border `accentBorder`, background `accentDim`. Both tiles press-animate to scale 0.96 with `withSpring`.
6. **Status card** (frosted glass): renders only when `jobState !== null`.
   - Score chips: two rounded pills side by side showing `score[0]` and `score[1]` in Barlow Condensed 28pt `#4ade80`.
   - Progress bar: thin (4dp) `#4ade80` bar with `accentDim` track.
   - Status label in `fontMono` 10pt `textMuted`.
   - Download button when `status === 'complete'`.
7. Bottom safe-area inset.

**State management:** unchanged from current `index.tsx` logic (upload, poll, results). Only the rendering layer changes.

---

## 4. Live Screen (`app/live.tsx`)

**Structure:**
1. `<ShuttleBackground />` — `position: absolute`, fills screen, z-index 0. Visible only when camera is not running; once `CameraView` is active, it sits behind the camera feed. Keep the background for the pre-session and post-session states.
2. `<CameraView>` — `StyleSheet.absoluteFillObject`, z-index 1, visible once session starts.
3. **Top overlay** (z-index 2): pinned to top safe-area, `position: absolute`.
   - Full-width frosted glass strip, `paddingHorizontal: 20`, `paddingVertical: 12`.
   - Score row: `score[0]` in Barlow Condensed 72pt `#4ade80` left, divider dot center, `score[1]` in Barlow Condensed 72pt `#fff` right.
   - Rally state label: `fontMono` 10pt `textMuted` letter-spacing 2, centered below scores.
4. **Calibrating banner**: amber pill (`#fbbf24` background, dark text), centered horizontally, slides in from top via `withSpring` when `status === 'calibrating'`, slides out when calibrated. Sits below the score strip.
5. **Bottom overlay** (z-index 2): pinned to bottom safe-area.
   - Left: connection dot (green `#4ade80` or red) + backlog counter in `fontMono` 10pt.
   - Center-right: circular stop button 64dp diameter, `destructive` red, Ionicons `stop` icon.
6. Pre-session state (before "Go live" tapped): shows the shuttle background full-screen with a centered frosted glass card containing a "Start session" button.
7. Post-session state (after stop): frosted card with final score and optional "Download annotated video" link.

---

## 5. Implementation Constraints

- **No new packages.** All animation via `react-native-reanimated` ~4.1.1. All SVG via `react-native-svg` 15.12.1. Gesture handling via `react-native-gesture-handler` ~2.28.0. All already in `package.json`.
- **Expo Go compatible.** No native modules, no dev build needed.
- **Faux glass only.** No `expo-blur`. The `rgba(255,255,255,0.07)` + `rgba(255,255,255,0.18)` border replicates the glass look shown in the approved demos.
- **Status bar.** Set to `dark-content` (light icons on dark background) via `expo-status-bar`.
- **Touch targets.** All interactive elements meet 44dp minimum per Apple HIG.
- **Surgical changes only.** Do not touch `backend/`, `session.py`, scoring pipeline, or any file outside `frontend/myApp/`.

---

## 6. Files Changed

| File | Change |
|---|---|
| `frontend/myApp/lib/theme.ts` | Full rewrite with new tokens |
| `frontend/myApp/components/ShuttleBackground.tsx` | New file |
| `frontend/myApp/app/index.tsx` | Rendering layer rewrite (logic unchanged) |
| `frontend/myApp/app/live.tsx` | Rendering layer rewrite (logic unchanged) |
| `frontend/myApp/app/_layout.tsx` | Set `StatusBar` to dark-content |

---

## 7. Out of Scope

- Dark/light toggle (dark only).
- Android-specific navigation bar coloring (deferred).
- Annotated video playback UI (deferred).
- Any backend changes.
