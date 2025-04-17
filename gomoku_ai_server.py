# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# 导入 ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import copy
import traceback # Import traceback for better error logging

# 如果是 PyInstaller 打包后，sys._MEIPASS 指向临时资源目录
BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder=None)
CORS(app)

# --- 常量 ---
SIZE   = 15
PLAYER = 1
AI     = 2
EMPTY  = 0

# 棋型打分（可以按需微调）
SCORE = {
    "FIVE":          100000000,
    "LIVE_FOUR":     10000000,
    "RUSH_FOUR":     500000,
    "LIVE_THREE":    50000,
    "SLEEP_THREE":   5000,
    "LIVE_TWO":      500,
    "SLEEP_TWO":     100,
    "CENTER_BONUS":  1 # 这个好像没用到，可以考虑加到评估里或移除
}

# Zobrist 哈希表，全局一次性初始化
zobrist_table = [[[random.getrandbits(64) for _ in range(3)]
                    for _ in range(SIZE)]
                   for _ in range(SIZE)]
empty_board_hash = 0 # 这个好像也没用到

# --- 公共工具函数 ---
def is_valid(board, x, y):
    # Simplified check assuming board size is constant SIZE
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    """判断 player 在 (x,y) 下子后是否五连"""
    if not (0 <= x < SIZE and 0 <= y < SIZE): return False # 先检查坐标有效性
    # Check only if the point is actually the player's piece - important!
    # The check might be called on an assumed move, so ensure the piece exists.
    if board[y][x] != player: return False # 确保该点是刚下的子 或 评估时假设的子

    directions = [(1,0),(0,1),(1,1),(1,-1)] # H, V, Diag\, Diag/
    for dx,dy in directions:
        cnt = 1
        # 正方向
        for i in range(1,5):
            nx,ny = x+i*dx, y+i*dy
            # Use direct bounds check, assuming SIZE is accessible or passed if needed
            if not (0 <= nx < SIZE and 0 <= ny < SIZE) or board[ny][nx] != player:
                break
            cnt+=1
        # 反方向
        for i in range(1,5):
            nx,ny = x-i*dx, y-i*dy
            if not (0 <= nx < SIZE and 0 <= ny < SIZE) or board[ny][nx] != player:
                break
            cnt+=1
        if cnt>=5:
            return True
    return False

# --- AI 核心 ---
class GomokuAI:

    def __init__(self, board, depth, max_workers=None):
        self.initial_board = [row[:] for row in board]
        self.depth = depth
        self.max_workers = max_workers if max_workers is not None else os.cpu_count() or 1
        # 限制最大工作单元数
        self.max_workers = max(1, min(self.max_workers, os.cpu_count() or 4))

    def _hash_board(self, board):
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                p = board[r][c]
                if p != EMPTY:
                    # Ensure zobrist_table is accessible
                    h ^= zobrist_table[r][c][p]
        return h

    def _update_hash(self, h, r, c, player):
         # Ensure zobrist_table is accessible
        return h ^ zobrist_table[r][c][player]

    def find_best_move(self):
        start_time = time.time()
        best_score = -math.inf
        best_move  = None
        board = [row[:] for row in self.initial_board] # Use a copy

        # 1. 先一步制胜？
        win1 = self._find_immediate_win(board, AI)
        if win1:
            print(f"[AI] 立即获胜于 {win1}")
            return {"x": win1[1], "y": win1[0]}

        # 2. 玩家一步制胜？去阻挡
        block1 = self._find_immediate_win(board, PLAYER)
        if block1:
            print(f"[AI] 立即阻止对手获胜于 {block1}")
            return {"x": block1[1], "y": block1[0]}

        # 3. Minimax+αβ+置换表+多进程
        print(f"[AI] 搜索深度 depth={self.depth} 并行工作单元数 max_workers={self.max_workers} (使用多进程)")
        moves = self._generate_moves(board, AI) # Generate moves for AI
        if not moves:
            print("[AI] 没有可选的移动了?")
            # Find any empty spot as fallback
            for rr in range(SIZE):
                for cc in range(SIZE):
                    if board[rr][cc]==EMPTY: return {"x":cc,"y":rr}
            return {"x":SIZE//2,"y":SIZE//2} # Should not happen on non-full board

        # 启发式排序
        scored = []
        for r,c in moves:
            board[r][c] = AI
            # Evaluate the board *after* placing the AI piece temporarily
            sc = self._evaluate_board(board, AI) # Evaluate from AI's perspective
            board[r][c] = EMPTY # Undo move
            scored.append(((r,c),sc))
        scored.sort(key=lambda x: x[1], reverse=True) # Sort descending by score
        sorted_moves = [mv for mv,_ in scored]
        print(f"[AI] 候选走法数量: {len(sorted_moves)}. 最佳初步评估走法: {sorted_moves[0]} 分数: {scored[0][1]:.1f}" if sorted_moves else "[AI] 无候选走法")

        # 置换表 - 每个进程将使用自己的独立表 (passed as argument)
        init_hash = self._hash_board(board)

        futures = []
        results = {} # 存储 future -> move 的映射

        # 使用 ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for r,c in sorted_moves:
                # 为每个任务创建棋盘副本
                b2 = [row[:] for row in board]
                b2[r][c] = AI # AI makes the move
                h2 = self._update_hash(init_hash, r, c, AI) # Update hash for the move

                # 提交任务给进程池: call _minimax_memo for the opponent's turn
                # Pass a *new empty dictionary* as the transposition table for each process
                fut = pool.submit(self._minimax_memo, # Target the instance method directly
                                  b2,                 # Board state *after* AI's move
                                  self.depth - 1,    # Remaining depth
                                  False,              # Now it's opponent's (minimizing) turn
                                  -math.inf,          # Initial alpha for this branch
                                  math.inf,           # Initial beta for this branch
                                  h2,                 # Hash after AI's move
                                  {})                # <<< FIX: Pass an empty dict as the table
                futures.append(fut)
                results[fut] = (r,c) # Associate future with the move (r, c)

            # 收集结果 (按提交顺序或完成顺序处理都可以)
            # 按提交顺序获取结果（更简单，便于调试）
            for i, fut in enumerate(futures):
                 move = results[fut]
                 try:
                     # The result 'sc' is the score *from the perspective of the AI*
                     # returned by the minimax call for the opponent's turn.
                     sc = fut.result() # Wait for the result (score)
                     print(f"[AI]   评估走法 {move} -> 分数: {sc:.1f}")
                     # AI wants to maximize this score
                     if sc > best_score:
                         best_score = sc
                         best_move = move
                 except Exception as e:
                     # Important: Log the error and the move that caused it
                     print(f"[AI] 子进程评估走法 {move} 时发生异常: {e}")
                     traceback.print_exc() # Print full traceback for debugging
                     # Fallback strategy: if the best move hasn't been set yet,
                     # and this was the evaluation for the heuristically best move, use it.
                     if best_move is None and i == 0:
                         print(f"[AI]   WARN: 首选评估失败，暂时选择 {move}")
                         best_move = move # Fallback to the first sorted move if its eval failed

        # 如果所有进程都失败了，或者没有找到任何有效分数(best_score is still -inf)
        # 选择启发式排序最好的那个
        if best_move is None:
            print("[AI] WARN: 所有子进程评估失败或无有效结果，选择初步评估最佳走法")
            if sorted_moves:
                best_move = sorted_moves[0]
            else:
                # Extremely unlikely fallback if even generation failed
                best_move = (SIZE//2, SIZE//2)
                # Ensure the chosen spot is empty if possible
                if board[best_move[0]][best_move[1]] != EMPTY:
                     for r_fall in range(SIZE):
                         for c_fall in range(SIZE):
                             if board[r_fall][c_fall] == EMPTY:
                                 best_move = (r_fall, c_fall)
                                 break
                         if board[best_move[0]][best_move[1]] == EMPTY: break


        # Ensure a move is selected, even if score remains -inf due to errors
        if best_move is None: # Should be handled above, but double-check
             best_move = (SIZE//2, SIZE//2) # Absolute fallback


        print(f"[AI] 选定走法 {best_move} 分数={best_score:.1f} 耗时 {(time.time()-start_time):.2f}s")
        # Return move in {"x": col, "y": row} format
        return {"x":best_move[1], "y":best_move[0]}


    # Removed the static _minimax_process_wrapper as it's no longer needed


    def _find_immediate_win(self, board, player):
        """Checks if 'player' can win in one move."""
        # Iterate through potential moves generated by _generate_moves
        # Use player here, could be AI or PLAYER depending on who we check for
        for r,c in self._generate_moves(board, player):
            if board[r][c]==EMPTY: # Only check empty spots
                board[r][c]=player # Try the move
                # check_win needs (board, player, x, y)
                won = check_win(board, player, c, r) # Check if this move wins
                board[r][c]=EMPTY # Undo the move
                if won:
                    return (r,c) # Return the winning move location
        return None # No immediate win found

    def _minimax_memo(self, board, depth, is_max, alpha, beta, h, table):
        """
        Minimax algorithm with Alpha-Beta Pruning and Transposition Table.
        'table' is the transposition table specific to this search branch/process.
        Returns the score from the perspective of the AI player.
        """
        # Use tuple (hash, depth, is_max) as key for transposition table
        key = (h, depth, is_max)
        if key in table:
            return table[key]

        # Check terminal states: Win/Loss/Draw (or max depth reached)
        # Use a quick check for existing win on the board *before* generating moves
        # This check needs to be efficient.
        winner = self._check_board_winner(board) # Check if someone *already* won
        if winner == AI:
             score = SCORE["FIVE"] * (depth + 1) # Faster win is better
             table[key] = score
             return score
        if winner == PLAYER:
             score = -SCORE["FIVE"] * (depth + 1) # Faster loss is worse
             table[key] = score
             return score
        # Consider draw? If no moves possible and no winner?

        if depth == 0:
            # Evaluate leaf node using the board evaluation function
            # Always evaluate from AI's perspective
            score = self._evaluate_board(board, AI)
            table[key]=score
            return score

        # Determine current player based on is_max flag
        current_player = AI if is_max else PLAYER
        moves = self._generate_moves(board, current_player)

        # If no moves possible from this state, evaluate the board
        if not moves:
            score = self._evaluate_board(board, AI)
            table[key]=score
            return score

        # --- Alpha-Beta Pruning Logic ---
        if is_max: # AI's turn (Maximizing player)
            best_val = -math.inf
            # Optional: Sort moves here as well for better pruning (can be costly)
            # sorted_moves = self._sort_moves(board, moves, AI)
            for r, c in moves: # or sorted_moves
                board[r][c] = AI # Make move
                h2 = self._update_hash(h, r, c, AI)
                val = self._minimax_memo(board, depth-1, False, alpha, beta, h2, table)
                board[r][c] = EMPTY # Undo move
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break # Beta cut-off
            table[key] = best_val
            return best_val
        else: # Player's turn (Minimizing player)
            best_val = math.inf
            # Optional: Sort moves for opponent
            # sorted_moves = self._sort_moves(board, moves, PLAYER)
            for r, c in moves: # or sorted_moves
                board[r][c] = PLAYER # Make move
                h2 = self._update_hash(h, r, c, PLAYER)
                val = self._minimax_memo(board, depth-1, True, alpha, beta, h2, table)
                board[r][c] = EMPTY # Undo move
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break # Alpha cut-off
            table[key] = best_val
            return best_val

    def _check_board_winner(self, board):
        """Checks if there is a winner currently on the board."""
        for r in range(SIZE):
            for c in range(SIZE):
                player = board[r][c]
                if player != EMPTY:
                    # If check_win confirms a win starting from (c, r) for this player
                    if check_win(board, player, c, r):
                        return player # Return the winner (AI or PLAYER)
        return EMPTY # No winner found

    def _generate_moves(self, board, player_to_move):
        """
        Generates potential moves, focusing on squares near existing pieces.
        """
        moves = set()
        has_piece = False
        radius = 1 # Consider neighbors within 1 step distance

        # Check for existing pieces and add neighbors
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c]!=EMPTY:
                    has_piece = True
                    # Add empty neighbors around this piece
                    for dr in range(-radius, radius+1):
                        for dc in range(-radius, radius+1):
                            if dr==0 and dc==0: continue
                            nr, nc = r+dr, c+dc
                            # Check bounds and if the neighbor spot is empty
                            if 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc]==EMPTY:
                                moves.add((nr,nc))

        # If the board is empty, play in the center
        if not has_piece:
            mid = SIZE//2
            return [(mid,mid)]

        # If no neighbors found (e.g., board full), return all empty cells (shouldn't normally happen mid-game)
        if not moves:
             return [(r,c) for r in range(SIZE) for c in range(SIZE) if board[r][c]==EMPTY]

        return list(moves)

    def _evaluate_board(self, board, who):
        """
        Evaluates the board state from the perspective of 'who' (usually AI).
        Positive score favors AI, negative favors Player.
        """
        # Calculate score for AI and Player separately
        my_score    = self._calculate_player_score(board, AI)
        oppo_score  = self._calculate_player_score(board, PLAYER)

        # Check for immediate win conditions (already handled by FIVE score, but can be explicit)
        if my_score   >= SCORE["FIVE"]: return SCORE["FIVE"] * 10 # Strong win signal
        if oppo_score >= SCORE["FIVE"]: return -SCORE["FIVE"] * 10 # Strong loss signal

        # Return AI score minus opponent score (opponent score weighted slightly higher)
        # This encourages both advancing AI's position and blocking the player.
        return my_score - oppo_score * 1.1

    def _calculate_player_score(self, board, player):
        """Calculates the total score for a given player based on patterns."""
        total_score = 0
        opponent = PLAYER if player == AI else AI

        # --- Evaluate all lines: Rows, Columns, Diagonals ---
        lines = []
        # Rows
        for r in range(SIZE):
            lines.append(board[r])
        # Columns
        for c in range(SIZE):
            lines.append([board[r][c] for r in range(SIZE)])
        # Diagonals (top-left to bottom-right)
        for i in range(-(SIZE - 5), SIZE - 4): # Optimized range for length >= 5
            line = []
            for j in range(SIZE):
                r, c = j, j + i
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(board[r][c])
            if len(line) >= 5: lines.append(line)
        # Diagonals (top-right to bottom-left)
        for i in range(4, SIZE * 2 - 5): # Optimized range for length >= 5
            line = []
            for j in range(SIZE):
                r, c = j, i - j
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(board[r][c])
            if len(line) >= 5: lines.append(line)

        # Evaluate each line for the player's patterns
        for line in lines:
            total_score += self._evaluate_line(line, player, opponent)

        return total_score

    def _evaluate_line(self, line, player, opponent):
        """Evaluates a single line (list) for the given player's patterns."""
        score = 0
        n = len(line)
        i = 0
        while i < n:
            # Find consecutive streaks of the player's pieces
            if line[i] == player:
                count = 0
                while i < n and line[i] == player:
                    count += 1
                    i += 1
                # After the streak, check open ends
                left_open = (i - count > 0 and line[i - count - 1] == EMPTY)
                right_open = (i < n and line[i] == EMPTY)
                open_ends = (1 if left_open else 0) + (1 if right_open else 0)
                score += self._score_pattern(count, open_ends)
            else:
                i += 1

        # --- Add checks for specific broken patterns (like P E P P, P P E P) ---
        # This part can significantly improve evaluation accuracy but adds complexity.
        # Example for P E P P -> Sleep Three potential
        for j in range(n - 3):
            if tuple(line[j:j+4]) == (player, EMPTY, player, player):
                 # Check ends for openness
                 left_empty = (j > 0 and line[j-1] == EMPTY)
                 right_empty = (j + 4 < n and line[j+4] == EMPTY)
                 if left_empty and right_empty:
                      score += SCORE["LIVE_THREE"] * 0.8 # High potential, slightly less than pure live three
                 elif left_empty or right_empty:
                      score += SCORE["SLEEP_THREE"] # Standard sleep three

        # Example for P P E P -> Sleep Three potential
        for j in range(n - 3):
            if tuple(line[j:j+4]) == (player, player, EMPTY, player):
                 left_empty = (j > 0 and line[j-1] == EMPTY)
                 right_empty = (j + 4 < n and line[j+4] == EMPTY)
                 if left_empty and right_empty:
                      score += SCORE["LIVE_THREE"] * 0.8
                 elif left_empty or right_empty:
                      score += SCORE["SLEEP_THREE"]

        # Add more patterns if needed (e.g., P E E P P, etc.)

        return score


    def _score_pattern(self, count, open_ends):
        """Assigns score based on piece count and open ends."""
        if count >= 5:
            return SCORE["FIVE"]

        # If both ends are blocked, the pattern is dead (unless it's already >= 5)
        if open_ends == 0 and count < 5:
            return 0

        # Score based on count and openness
        if count == 4:
            if open_ends == 2: return SCORE["LIVE_FOUR"]
            if open_ends == 1: return SCORE["RUSH_FOUR"]
        if count == 3:
            if open_ends == 2: return SCORE["LIVE_THREE"]
            if open_ends == 1: return SCORE["SLEEP_THREE"]
        if count == 2:
            if open_ends == 2: return SCORE["LIVE_TWO"]
            if open_ends == 1: return SCORE["SLEEP_TWO"]
        # Optional: score for single pieces (usually low value)
        # if count == 1:
        #    if open_ends == 2: return 10 # e.g., LIVE_ONE
        #    if open_ends == 1: return 1  # e.g., SLEEP_ONE

        return 0 # Default for counts < 2 or unhandled cases


# --- Flask 路由 ---
@app.route('/', methods=['GET'])
def index():
    # Serve the main HTML file from the base directory
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json or {}
    if 'board' not in data or 'depth' not in data:
        return jsonify({"error":"Missing 'board' or 'depth' parameter"}), 400

    board = data['board']
    depth = int(data['depth'])
    # Use 'threads' from frontend, map to 'workers' internally
    workers = int(data.get('threads', os.cpu_count() or 1))

    # --- Input Validation ---
    if not (1 <= depth <= 12): # Depth limit
        return jsonify({"error":"Search depth (depth) must be between 1 and 12"}), 400
    if not (1 <= workers <= (os.cpu_count() or 4) * 2): # More flexible worker limit, but warn if high
         print(f"WARN: Requested workers ({workers}) exceeds typical recommendations.")
        # return jsonify({"error":"Parallel workers (threads) must be between 1 and 100"}), 400 # Example limit
    if not (isinstance(board, list) and len(board) == SIZE and
            all(isinstance(row, list) and len(row) == SIZE for row in board) and
            all(cell in [EMPTY, PLAYER, AI] for row in board for cell in row)):
        return jsonify({"error":"Board format is invalid or contains illegal values"}), 400

    # --- Create AI instance and get move ---
    print(f"Received request: depth={depth}, workers={workers}") # Log request details
    ai = GomokuAI(board, depth, workers)
    try:
        move = ai.find_best_move() # This now handles the fixed multiprocessing
        if move is None:
             print("ERROR: AI failed to determine a move.")
             # Provide a fallback move if AI returns None unexpectedly
             # Find any empty cell
             fallback_move = None
             for r in range(SIZE):
                 for c in range(SIZE):
                     if board[r][c] == EMPTY:
                         fallback_move = {"x": c, "y": r}
                         break
                 if fallback_move: break
             if fallback_move:
                 return jsonify({"move": fallback_move, "warning": "AI failed, used fallback"})
             else: # Board must be full?
                 return jsonify({"error":"AI failed and no empty cells found (Draw?)"}), 500

        print(f"Sending move: {move}") # Log the move being sent
        return jsonify({"move": move})

    except Exception as e:
        # Log the detailed error on the server side
        print("="*20 + " AI Calculation Error " + "="*20)
        print(f"Request Data: depth={depth}, workers={workers}") # Board might be too large to log cleanly
        # print(f"Board state: {board}") # Optional: log board if needed, can be large
        traceback.print_exc() # Print full stack trace
        print("="*50)
        # Return a generic error message to the client
        return jsonify({"error": f"AI internal calculation error: {type(e).__name__}"}), 500


# --- 启动 ---
if __name__=='__main__':
    # Crucial for multiprocessing when frozen (e.g., with PyInstaller)
    # Should be called early in the main entry point.
    # Consider moving freeze_support() to main.py if that's the actual entry point.
    # multiprocessing.freeze_support() # Usually needed in main.py's if __name__ == '__main__':

    port = 5000
    print(f"=== Gomoku AI Server (Multiprocessing - Fixed) starting on port {port} ===")
    print(f"Detected CPU cores: {os.cpu_count() or 'Unknown'}")
    # Run with Flask's built-in server - disable debug and reloader for stability with multiprocessing
    # threaded=False is important as ProcessPoolExecutor handles parallelism.
    app.run(host='127.0.0.1', port=port, debug=False, threaded=False, use_reloader=False)