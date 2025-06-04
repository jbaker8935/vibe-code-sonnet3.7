// Overlay and modal management
export function showOverlay(overlay) {
    if (overlay) overlay.classList.add('active');
}
export function hideOverlay(overlay) {
    if (overlay) overlay.classList.remove('active');
}
export function hideAllOverlays() {
    document.querySelectorAll('.overlay').forEach(overlay => overlay.classList.remove('active'));
}

// Example: showOverlay, hideOverlay, hideAllOverlays, etc.
// (Implementations would be moved here from script.js)

// Export all overlay functions needed by main.js

// ...
