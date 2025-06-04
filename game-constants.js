// Game constants and shared enums
export const ROWS = 8;
export const COLS = 4;
export const PLAYER_A = 'A'; // White
export const PLAYER_B = 'B'; // Black
export const NORMAL = 'normal';
export const SWAPPED = 'swapped';
export const NUM_DIRECTIONS = 8;
export const JS_DIRECTIONS = [
    { dr: -1, dc: -1 }, // 0
    { dr: -1, dc:  0 }, // 1
    { dr: -1, dc:  1 }, // 2
    { dr:  0, dc: -1 }, // 3
    { dr:  0, dc:  1 }, // 4
    { dr:  1, dc: -1 }, // 5
    { dr:  1, dc:  0 }, // 6
    { dr:  1, dc:  1 }  // 7
];

export const initialPosition = [
    "BBBB\nBBBB\n....\n....\n....\n....\nAAAA\nAAAA",
    "....\n....\nBBBB\nBBBB\nAAAA\nAAAA\n....\n....",
    "BB..\nBB..\nBB..\nBB..\n..AA\n..AA\n..AA\n..AA",
    "B...\nBB..\nBB..\nBBB.\n.AAA\n..AA\n..AA\n...A",
    "B..B\n.BB.\n.BB.\nB..B\nA..A\n.AA.\n.AA.\nA..A",
    "....\n....\nBABA\nABAB\nBABA\nABAB\n....\n...."        
];
