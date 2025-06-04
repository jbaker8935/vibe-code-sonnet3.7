// Rendering functions and DOM element references
import { PLAYER_A, PLAYER_B, NORMAL, SWAPPED, ROWS, COLS } from './game-constants.js';

export function getPieceImage(player, state) {
    const color = player === PLAYER_A ? 'white' : 'black';
    const type = state === NORMAL ? 'normal' : 'swapped';
    return `images/${color}_${type}.png`;
}

export function renderBoard({
    board,
    selectedPiece,
    legalMoves,
    gameOver,
    winner,
    winPath,
    moveHistory,
    currentHistoryIndex,
    playerAScore,
    playerBScore,
    boardElement,
    currentPlayer,
    showWinPath // PATCH: allow controlling win path highlight
}) {
    boardElement.innerHTML = '';
    boardElement.classList.remove('game-over');
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.classList.add(((r + c) % 2 === 0) ? 'light' : 'dark');
            const pieceData = board[r][c];
            if (pieceData) {
                const pieceImg = document.createElement('img');
                pieceImg.classList.add('piece');
                pieceImg.src = getPieceImage(pieceData.player, pieceData.state);
                pieceImg.alt = `Player ${pieceData.player} ${pieceData.state}`;
                cell.appendChild(pieceImg);
            }
            if (selectedPiece && selectedPiece.row === r && selectedPiece.col === c) {
                cell.classList.add('selected');
            }
            if (legalMoves && legalMoves.some(move => move.row === r && move.col === c)) {
                cell.classList.add('legal-move');
                if (board[r][c] && board[r][c].player !== currentPlayer) {
                    cell.classList.add('swap-target');
                }
            }
            // PATCH: Only highlight win path if showWinPath is true
            if (showWinPath && gameOver && winner && winPath && winPath.some(pos => pos.row === r && pos.col === c)) {
                cell.classList.add('win-path');
            }
            boardElement.appendChild(cell);
        }
    }
    if (gameOver) {
        boardElement.classList.add('game-over');
        if (currentHistoryIndex !== undefined) {
            const moveCount = document.createElement('div');
            moveCount.classList.add('move-counter');
            moveCount.textContent = currentHistoryIndex === moveHistory.length ?
                'Final Position' :
                `Move ${currentHistoryIndex} of ${moveHistory.length}`;
            boardElement.appendChild(moveCount);
        }
        const scoreDisplay = document.createElement('div');
        scoreDisplay.classList.add('score-display');
        scoreDisplay.innerHTML = `
            <div id="score-a">Player A: ${playerAScore}</div>
            <div id="score-b">Player B: ${playerBScore}</div>
        `;
        boardElement.appendChild(scoreDisplay);
    }
}

// Export DOM element getters for use in main.js
export function getBoardElement() {
    return document.getElementById('game-board');
}
export function getScoreElements() {
    return {
        controlScoreA: document.getElementById('control-score-a'),
        controlScoreB: document.getElementById('control-score-b'),
        scoreA: document.getElementById('score-a'),
        scoreB: document.getElementById('score-b'),
    };
}

// ...other rendering and UI update functions can be added here...
