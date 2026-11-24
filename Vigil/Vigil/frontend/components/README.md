# Standalone VisionOS Prism Capsule & Liquid Glass Optical Lens Component

This folder contains the complete, zero-dependency **Prism Capsule / Liquid Glass Optical Lens** component extracted from VIGIL. You can drop this component into any web project (React, Next.js, Vue, Svelte, or Vanilla HTML/CSS).

---

## Files Included

1. `index.html` / `prism-capsule-standalone.html`:
   - Standalone demo with Three.js ambient background scene.
   - Interactive draggable glass lens with pointer physics.
   - Dynamic SVG Signed-Distance-Field (SDF) displacement generator for realistic optical lens curvature.
   - Keyboard shortcut `[L]` and toggle button integration.

---

## How It Works

### 1. HTML Element
Place this container anywhere in your `<body>`:
```html
<div class="liquid-glass-lens" id="hero-lens" data-name="Prism Capsule" title="Drag to refract background">
    <div class="lens-badge">PRISM CAPSULE &bull; OPTICAL LENS</div>
</div>
```

### 2. Required CSS
```css
.liquid-glass-lens {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: clamp(340px, 46vw, 560px);
    height: clamp(170px, 24vh, 270px);
    border-radius: 9999px;
    z-index: 50;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
    cursor: grab;
    pointer-events: auto;
    backdrop-filter: url(#liquid-convex-lens) blur(1px) saturate(100%);
    -webkit-backdrop-filter: url(#liquid-convex-lens) blur(1px) saturate(100%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    box-shadow: 
        0 30px 80px -15px rgba(0, 0, 0, 0.8),
        inset 0 1px 0 0 rgba(255, 255, 255, 0.2),
        inset 0 -1px 0 0 rgba(0, 0, 0, 0.6);
}

.liquid-glass-lens:active {
    cursor: grabbing;
}

.liquid-glass-lens.hidden {
    display: none !important;
}

.lens-badge {
    font-family: monospace;
    font-size: 0.68rem;
    color: #FFFFFF;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    padding: 0.45rem 1.15rem;
    background: rgba(0, 0, 0, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    backdrop-filter: blur(12px);
    pointer-events: none;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.8);
}
```

### 3. JavaScript Engine Initialization
```javascript
const lensEngine = new LiquidGlassEngine({
    target: '#hero-lens',
    filterId: 'liquid-convex-lens',
    shape: 'capsule',
    displacementScale: 0.16,
    bevelRatio: 0.55,
    draggable: true
});
```

