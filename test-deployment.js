#!/usr/bin/env node

/**
 * Simple test script to verify the deployment bundle works
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function testBundle() {
    const bundlePath = path.resolve(__dirname, 'dist', 'bundle.js');
    
    if (!fs.existsSync(bundlePath)) {
        console.error('❌ Bundle not found. Run "node deploy.js" first.');
        process.exit(1);
    }
    
    const bundleContent = fs.readFileSync(bundlePath, 'utf-8');
    
    // Check for essential functions
    const requiredFunctions = [
        'window.parseStartingPosition',
        'window.renderBoard', 
        'window.checkWinCondition',
        'window.findBestAIMove'
    ];
    
    const missingFunctions = requiredFunctions.filter(fn => !bundleContent.includes(fn));
    
    if (missingFunctions.length > 0) {
        console.error('❌ Missing functions in bundle:', missingFunctions);
        process.exit(1);
    }
    
    // Check for essential constants
    const requiredConstants = [
        'window.ROWS = 8',
        'window.COLS = 4', 
        'window.PLAYER_A',
        'window.PLAYER_B'
    ];
    
    const missingConstants = requiredConstants.filter(constant => !bundleContent.includes(constant));
    
    if (missingConstants.length > 0) {
        console.error('❌ Missing constants in bundle:', missingConstants);
        process.exit(1);
    }
    
    // Check file sizes
    const bundleSize = Math.round(bundleContent.length / 1024);
    console.log(`✅ Bundle test passed!`);
    console.log(`   Bundle size: ${bundleSize}KB`);
    console.log(`   Functions: ${requiredFunctions.length} found`);
    console.log(`   Constants: ${requiredConstants.length} found`);
    
    // Check if all required files exist
    const distDir = path.resolve(__dirname, 'dist');
    const requiredFiles = [
        'index.html',
        'style.css',
        'bundle.js',
        'game_logic_adapter.js',
        'mcts_js.js'
    ];
    
    const missingFiles = requiredFiles.filter(file => !fs.existsSync(path.join(distDir, file)));
    
    if (missingFiles.length > 0) {
        console.error('❌ Missing files in dist:', missingFiles);
        process.exit(1);
    }
    
    console.log(`✅ All required files present: ${requiredFiles.join(', ')}`);
    
    // Check if directories exist
    const requiredDirs = ['images', 'switcharoo_tfjs_model'];
    const missingDirs = requiredDirs.filter(dir => !fs.existsSync(path.join(distDir, dir)));
    
    if (missingDirs.length > 0) {
        console.error('❌ Missing directories in dist:', missingDirs);
        process.exit(1);
    }
    
    console.log(`✅ All required directories present: ${requiredDirs.join(', ')}`);
    console.log(`\n🎉 Deployment test successful! Ready for production.`);
}

testBundle();
