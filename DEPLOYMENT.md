# Switcharoo Game Deployment

This document explains how to create a deployment bundle for the Switcharoo game that can be served by any static web server.

## Quick Start

```bash
# Development build (includes debug logs)
node deploy.js

# Production build (optimized, no debug logs)
node deploy.js --production

# Custom output directory
node deploy.js build --production
```

## What the Script Does

The `deploy.js` script creates a self-contained deployment bundle by:

1. **Bundling ES Modules**: Combines all JavaScript modules (`main.js`, `game-*.js`) into a single `bundle.js` file
2. **Converting Imports/Exports**: Transforms ES6 module syntax to browser-compatible global variables
3. **Copying Assets**: Includes all necessary files (CSS, images, TensorFlow.js model, standalone scripts)
4. **Updating HTML**: Modifies `index.html` to reference the bundled script instead of ES modules
5. **Production Optimization**: Removes debug console.log statements in production mode

## Output Structure

After running the deployment script, the `dist/` folder will contain:

```
dist/
├── index.html              # Modified to use bundle.js
├── bundle.js               # All ES modules combined (92KB production, 102KB dev)
├── style.css               # Game styles
├── game_logic_adapter.js   # Standalone MCTS adapter
├── mcts_js.js             # MCTS implementation
├── images/                # All game icons and graphics
└── switcharoo_tfjs_model/ # TensorFlow.js neural network model
```

## Module Dependencies

The bundler processes these modules in dependency order:

1. `game-constants.js` - Game constants and enums
2. `game-board.js` - Board manipulation functions  
3. `game-render.js` - DOM rendering functions
4. `game-logic.js` - Core game logic
5. `game-ai.js` - Basic AI and neural network interface
6. `game-mcts-wrapper.js` - MCTS integration wrapper
7. `game-overlays.js` - UI overlay management
8. `game-ai-advanced.js` - Advanced AI algorithms
9. `test-board-position.js` - Testing utilities
10. `main.js` - Main entry point and initialization

## Testing the Deployment

Use the included test script to verify the bundle:

```bash
node test-deployment.js
```

This checks for:
- ✅ Required functions are exported
- ✅ Essential constants are defined  
- ✅ All files are present
- ✅ Proper bundle size

## Serving the Deployment

### Python Built-in Server
```bash
cd dist
python -m http.server 8000
# Open http://localhost:8000
```

### Node.js
```bash
cd dist
npx serve .
# Or: npx http-server .
```

### Apache/Nginx
Simply copy the `dist/` contents to your web server's document root.

## External Dependencies

The deployment uses these CDN resources (loaded by `index.html`):
- **TensorFlow.js Core**: `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js`
- **TensorFlow.js WASM Backend**: For better mobile performance

These are loaded before the bundled scripts to ensure TensorFlow.js is available.

## Production vs Development

| Feature | Development | Production |
|---------|------------|------------|
| Bundle Size | 102KB | 92KB |
| Debug Logs | Included | Removed |
| Build Time | Fast | Fast |
| Optimization | None | Console.log removal |

## Browser Compatibility

The bundled deployment works in all modern browsers that support:
- ES6 (ECMAScript 2015)
- WebGL (for TensorFlow.js)
- Canvas API
- Modern CSS

## Troubleshooting

### Bundle Test Fails
```bash
# Rebuild and test
rm -rf dist
node deploy.js --production
node test-deployment.js
```

### Game Won't Load
1. Check browser console for errors
2. Ensure TensorFlow.js CDN is accessible
3. Verify all files are served correctly (no 404s)
4. Test with a simple local server

### Performance Issues
- Use production build (`--production` flag)
- Ensure TensorFlow.js WASM backend loads
- Check Network tab for large resource downloads

## File Size Breakdown

- **bundle.js**: 92KB (production) - All game logic
- **switcharoo_tfjs_model/**: ~2MB - Neural network weights
- **images/**: ~50KB - SVG icons and graphics
- **mcts_js.js**: ~15KB - Monte Carlo Tree Search
- **game_logic_adapter.js**: ~5KB - MCTS interface
- **style.css**: ~10KB - Game styling

**Total**: ~2.2MB (mostly the neural network model)

## Customization

### Adding New Modules
1. Add module filename to `ES_MODULES` array in `deploy.js`
2. Ensure proper dependency order
3. Update imports in dependent modules

### Changing Build Output
Modify the `ASSETS_TO_COPY` array to include/exclude files and directories.

### Advanced Optimization
For further size reduction, consider:
- Minifying CSS
- Optimizing images  
- Compressing the TensorFlow.js model
- Using a dedicated bundler (Webpack, Rollup, Vite)
