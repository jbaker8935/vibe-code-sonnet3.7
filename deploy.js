#!/usr/bin/env node

/**
 * Deployment script for Switcharoo game
 * 
 * This script creates a bundled version of main.js and all its dependencies
 * that can be used with index.html for deployment.
 * 
 * Usage: 
 *   node deploy.js [output-dir] [--production]
 *   node deploy.js                    # Development build to ./dist
 *   node deploy.js build              # Development build to ./build  
 *   node deploy.js --production       # Production build to ./dist (minified, no debug logs)
 *   node deploy.js dist --production  # Production build to ./dist
 * 
 * Features:
 *   - Bundles all ES modules into a single file
 *   - Converts ES6 imports/exports to browser-compatible code
 *   - Copies all assets (CSS, images, models, standalone scripts)
 *   - Updates index.html to use bundled script
 *   - Optional production mode with debug log removal
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { minify } from 'terser';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const args = process.argv.slice(2);
const isProduction = args.includes('--production');
const outputDirArg = args.find(arg => !arg.startsWith('--'));
const OUTPUT_DIR = outputDirArg || 'dist';
const ENTRY_FILE = 'main.js';

// List of ES modules that need to be bundled (in dependency order)
const ES_MODULES = [
    'game-constants.js',
    'game-board.js',
    'game-render.js',
    'game-logic.js',
    'game-ai.js',
    'game-mcts-wrapper.js', 
    'game-overlays.js',
    'game-ai-advanced.js',
    'test-board-position.js',
    'main.js'
];

// Standalone scripts that need to be copied as-is (not ES modules)
const STANDALONE_SCRIPTS = [
    'game_logic_adapter.js',
    'mcts_js.js'
];

// Assets and other files to copy
const ASSETS_TO_COPY = [
    'index.html',
    'style.css',
    'images/',
    'switcharoo_tfjs_model/',
    'favicon.ico'
];

function log(message) {
    console.log(`[DEPLOY] ${message}`);
}

function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        log(`Created directory: ${dirPath}`);
    }
}

function copyFile(src, dest) {
    const srcPath = path.resolve(__dirname, src);
    const destPath = path.resolve(__dirname, OUTPUT_DIR, dest);
    
    if (!fs.existsSync(srcPath)) {
        log(`Warning: Source file not found: ${src}`);
        return false;
    }
    
    ensureDir(path.dirname(destPath));
    fs.copyFileSync(srcPath, destPath);
    log(`Copied: ${src} -> ${dest}`);
    return true;
}

function copyDirectory(src, dest) {
    const srcPath = path.resolve(__dirname, src);
    const destPath = path.resolve(__dirname, OUTPUT_DIR, dest);
    
    if (!fs.existsSync(srcPath)) {
        log(`Warning: Source directory not found: ${src}`);
        return false;
    }
    
    ensureDir(destPath);
    
    const items = fs.readdirSync(srcPath, { withFileTypes: true });
    for (const item of items) {
        const srcItem = path.join(srcPath, item.name);
        const destItem = path.join(destPath, item.name);
        
        if (item.isDirectory()) {
            copyDirectory(path.join(src, item.name), path.join(dest, item.name));
        } else {
            fs.copyFileSync(srcItem, destItem);
        }
    }
    
    log(`Copied directory: ${src} -> ${dest}`);
    return true;
}

function processModule(filePath) {
    log(`Processing module: ${filePath}`);
    
    const fullPath = path.resolve(__dirname, filePath);
    if (!fs.existsSync(fullPath)) {
        log(`Warning: Module not found: ${filePath}`);
        return '';
    }
    
    let content = fs.readFileSync(fullPath, 'utf-8');
    
    // Remove import statements (we'll inline everything)
    // Handle both single-line and multi-line imports
    content = content.replace(/^import\s+[\s\S]*?from\s+['"][^'"]*['"];?\s*$/gm, '');
    
    // Convert export statements to window assignments for global access
    content = content.replace(/^export\s+function\s+(\w+)/gm, 'window.$1 = function $1');
    content = content.replace(/^export\s+async\s+function\s+(\w+)/gm, 'window.$1 = async function $1');
    content = content.replace(/^export\s+const\s+(\w+)\s*=/gm, 'window.$1 =');
    content = content.replace(/^export\s+let\s+(\w+)\s*=/gm, 'window.$1 =');
    content = content.replace(/^export\s+\{([^}]+)\}/gm, (match, exports) => {
        // Handle export { name1, name2, ... }
        const names = exports.split(',').map(name => name.trim());
        return names.map(name => `window.${name} = ${name};`).join('\n');
    });
    
    // Remove console.log statements in production mode
    if (isProduction) {
        content = content.replace(/^\s*console\.log\(.*?\);?\s*$/gm, '');
    }
    
    // Add module separator comment
    const separator = `\n// ========== ${filePath} ==========\n`;
    
    return separator + content + '\n';
}

async function minifyJavaScript(code) {
    try {
        const result = await minify(code, {
            compress: {
                drop_console: false, // We already handle console.log removal separately
                drop_debugger: true,
                dead_code: true,
                unused: true,
                conditionals: true,
                evaluate: true,
                booleans: true,
                loops: true,
                if_return: true,
                join_vars: true,
                collapse_vars: true,
                reduce_vars: true,
                warnings: false,
                pure_getters: true,
                unsafe: false,
                unsafe_comps: false,
                side_effects: false
            },
            mangle: {
                // Keep some function names for better debugging
                keep_fnames: false,
                reserved: ['window', 'document', 'console', 'tf', 'MCTSSearch', 'SwitcharooGameLogic']
            },
            format: {
                comments: false,
                beautify: false,
                semicolons: true
            }
        });
        
        if (result.error) {
            throw result.error;
        }
        
        return result.code;
    } catch (error) {
        log(`Warning: Minification failed, using unminified code. Error: ${error.message}`);
        return code;
    }
}

async function createBundle() {
    log('Creating JavaScript bundle...');
    
    let bundleContent = `// Switcharoo Game Bundle
// Generated on: ${new Date().toISOString()}
// 
// This file contains all ES modules bundled together for deployment.
// Original modules: ${ES_MODULES.join(', ')}

(function() {
    'use strict';
    
`;

    // Process each module in dependency order
    for (const module of ES_MODULES) {
        bundleContent += processModule(module);
    }
    
    // Add initialization code - main.js DOMContentLoaded is already included
    bundleContent += `
})();
`;

    // Minify the bundle in production mode
    if (isProduction) {
        log('Minifying JavaScript bundle...');
        bundleContent = await minifyJavaScript(bundleContent);
    }

    // Write the bundle
    const bundlePath = path.resolve(__dirname, OUTPUT_DIR, 'bundle.js');
    ensureDir(path.dirname(bundlePath));
    fs.writeFileSync(bundlePath, bundleContent);
    log(`Bundle created: bundle.js (${Math.round(bundleContent.length / 1024)}KB)`);
}

function createBundledIndex() {
    log('Creating bundled index.html...');
    
    const indexPath = path.resolve(__dirname, 'index.html');
    if (!fs.existsSync(indexPath)) {
        log('Error: index.html not found');
        return;
    }
    
    let indexContent = fs.readFileSync(indexPath, 'utf-8');
    
    // Replace the module script with bundle script
    indexContent = indexContent.replace(
        /<script type="module" src="main\.js"><\/script>/,
        '<script src="bundle.js"></script>'
    );
    
    // Update paths for standalone scripts if they exist
    for (const script of STANDALONE_SCRIPTS) {
        const regex = new RegExp(`src="${script}"`, 'g');
        indexContent = indexContent.replace(regex, `src="${script}"`);
    }
    
    // Write the bundled index.html
    const outputPath = path.resolve(__dirname, OUTPUT_DIR, 'index.html');
    fs.writeFileSync(outputPath, indexContent);
    log('Created bundled index.html');
}

async function main() {
    const buildMode = isProduction ? 'production' : 'development';
    
    log(`Starting ${buildMode} deployment to: ${OUTPUT_DIR}`);
    log(`Working directory: ${__dirname}`);
    
    if (isProduction) {
        log('Production mode: Removing debug logs and optimizing bundle');
    }
    
    // Ensure output directory exists
    ensureDir(path.resolve(__dirname, OUTPUT_DIR));
    
    // Create JavaScript bundle
    await createBundle();
    
    // Copy standalone scripts
    for (const script of STANDALONE_SCRIPTS) {
        copyFile(script, script);
    }
    
    // Copy assets
    for (const asset of ASSETS_TO_COPY) {
        if (asset === 'index.html') {
            // Handle index.html specially to update script references
            createBundledIndex();
        } else if (asset.endsWith('/')) {
            // Directory
            const dirName = asset.slice(0, -1);
            copyDirectory(dirName, dirName);
        } else {
            // Regular file
            copyFile(asset, asset);
        }
    }
    
    log(`\n${buildMode.toUpperCase()} deployment complete!`);
    log(`Files are ready in: ${path.resolve(__dirname, OUTPUT_DIR)}/`);
    log(`\nTo test the deployment:`);
    log(`  cd ${OUTPUT_DIR}`);
    log(`  python -m http.server 8000`);
    log(`  # Then open http://localhost:8000 in your browser`);
    log(`\nBundle size: ${Math.round(fs.statSync(path.resolve(__dirname, OUTPUT_DIR, 'bundle.js')).size / 1024)}KB`);
}

// Run the deployment
main();
