# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from concurrent.futures import ProcessPoolExecutor
import copy
import traceback # Import traceback for better error logging

# --- Numba and NumPy Imports ---
import numpy as np
import numba

# --- Constants ---
SIZE   = 30 # Board size remains 30x30
PLAYER = np.int8(1)
AI     = np.int8(2)
EMPTY  = np.int8(0)

BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder=None)
CORS(app)

# --- Score constants ---
SCORE_FIVE = 100000000
SCORE_LIVE_FOUR = 10000000
SCORE_RUSH_FOUR = 500000
SCORE_LIVE_THREE = 50000
SCORE_SLEEP_THREE = 5000
SCORE_LIVE_TWO = 500
SCORE_SLEEP_TWO = 100

# Zobrist Hash Table (uses SIZE=30)
zobrist_table = [[[random.getrandbits(64) for _ in range(3)]
                    for _ in range(SIZE)]
                   for _ in range(SIZE)]

# --- Numba Accelerated Helper Functions (No changes needed) ---

@numba.njit(cache=True)
def _check_win_numba(board_arr: np.ndarray, player: np.int8, x: int, y: int, board_size: int) -> bool:
    """Numba-accelerated win check."""
    if not (0 <= y < board_size and 0 <= x < board_size and board_arr[y, x] == player):
         return False
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for dx, dy in directions:
        cnt = 1
        for sign in numba.int64([-1, 1]): # Check both directions using Numba typed loop
            for i in range(1, 5):
                nx, ny = x + i * dx * sign, y + i * dy * sign
                if not (0 <= nx < board_size and 0 <= ny < board_size and board_arr[ny, nx] == player):
                    break
                cnt += 1
        if cnt >= 5:
            return True
    return False

@numba.njit(cache=True)
def _score_pattern_numba(count: int, open_ends: int,
                         s_five: int, s_live4: int, s_rush4: int,
                         s_live3: int, s_sleep3: int, s_live2: int, s_sleep2: int) -> int:
    """Numba-accelerated pattern scoring helper."""
    if count >= 5: return s_five
    if open_ends == 0 and count < 5: return 0
    if count == 4: return s_live4 if open_ends == 2 else s_rush4
    if count == 3: return s_live3 if open_ends == 2 else s_sleep3
    if count == 2: return s_live2 if open_ends == 2 else s_sleep2
    return 0

@numba.njit(cache=True)
def _evaluate_line_numba(line: np.ndarray, player: np.int8, opponent: np.int8,
                         s_five: int, s_live4: int, s_rush4: int,
                         s_live3: int, s_sleep3: int, s_live2: int, s_sleep2: int) -> int:
    """Numba-accelerated line evaluation."""
    score = 0
    n = len(line)
    i = 0
    while i < n:
        if line[i] == player:
            count = 0
            start_idx = i
            while i < n and line[i] == player: count += 1; i += 1
            left_open = (start_idx > 0 and line[start_idx - 1] == EMPTY)
            right_open = (i < n and line[i] == EMPTY)
            open_ends = (1 if left_open else 0) + (1 if right_open else 0)
            score += _score_pattern_numba(count, open_ends, s_five, s_live4, s_rush4, s_live3, s_sleep3, s_live2, s_sleep2)
        else: i += 1

    # Broken patterns
    for j in range(n - 3):
        # P E P P
        if line[j] == player and line[j+1] == EMPTY and line[j+2] == player and line[j+3] == player:
            left_empty = (j > 0 and line[j-1] == EMPTY)
            right_empty = (j + 4 < n and line[j+4] == EMPTY)
            if left_empty and right_empty: score += int(s_live3 * 0.8)
            elif left_empty or right_empty: score += s_sleep3
        # P P E P
        elif line[j] == player and line[j+1] == player and line[j+2] == EMPTY and line[j+3] == player:
            left_empty = (j > 0 and line[j-1] == EMPTY)
            right_empty = (j + 4 < n and line[j+4] == EMPTY)
            if left_empty and right_empty: score += int(s_live3 * 0.8)
            elif left_empty or right_empty: score += s_sleep3
    return score

@numba.njit(cache=True)
def _calculate_player_score_numba(board_arr: np.ndarray, player: np.int8, opponent: np.int8,
                                  board_size: int,
                                  s_five: int, s_live4: int, s_rush4: int,
                                  s_live3: int, s_sleep3: int, s_live2: int, s_sleep2: int) -> int:
    """Numba-accelerated score calculation for a player."""
    total_score = 0
    scores = (s_five, s_live4, s_rush4, s_live3, s_sleep3, s_live2, s_sleep2)

    # Rows & Columns
    for i in range(board_size):
        total_score += _evaluate_line_numba(board_arr[i, :], player, opponent, *scores)
        total_score += _evaluate_line_numba(board_arr[:, i], player, opponent, *scores)

    # Diagonals
    flipped_board = np.fliplr(board_arr) # Pre-flip for anti-diagonals
    for k in range(-(board_size - 5), board_size - 4):
        diag = np.diag(board_arr, k=k)
        anti_diag = np.diag(flipped_board, k=k)
        if len(diag) >= 5: # np.diag might return shorter arrays for corner diagonals
             total_score += _evaluate_line_numba(diag, player, opponent, *scores)
        if len(anti_diag) >= 5:
             total_score += _evaluate_line_numba(anti_diag, player, opponent, *scores)

    return total_score

# --- Public Utility Functions (No changes needed) ---
def check_win(board, player, x, y):
    """Public wrapper for win check"""
    if not (0 <= x < SIZE and 0 <= y < SIZE): return False
    board_arr = np.array(board, dtype=np.int8)
    player_np = np.int8(player)
    return _check_win_numba(board_arr, player_np, x, y, SIZE)

# --- AI Core Class (No major logic changes needed) ---
class GomokuAI:
    def __init__(self, board, depth, max_workers=None):
        self.initial_board_list = [row[:] for row in board]
        self.depth = depth
        self.max_workers = max_workers if max_workers is not None else os.cpu_count() or 1
        self.max_workers = max(1, min(self.max_workers, os.cpu_count() * 4 or 8)) # Allow more workers
        self.score_constants = (
            SCORE_FIVE, SCORE_LIVE_FOUR, SCORE_RUSH_FOUR,
            SCORE_LIVE_THREE, SCORE_SLEEP_THREE, SCORE_LIVE_TWO, SCORE_SLEEP_TWO
        )
        self.transposition_table = {} # Instance-level table for multi-processing safety

    def _hash_board(self, board_list):
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                p = board_list[r][c] # Should be 0, 1, or 2
                if p != EMPTY: h ^= zobrist_table[r][c][p]
        return h

    def _update_hash(self, h, r, c, player):
        return h ^ zobrist_table[r][c][player]

    def find_best_move(self):
        start_time = time.time()
        best_score = -math.inf
        best_move = None
        board_list = [row[:] for row in self.initial_board_list]

        # 1. Check immediate win/block
        win1 = self._find_immediate_win(board_list, AI)
        if win1: print(f"[AI] Immediate win at {win1}"); return {"x": win1[1], "y": win1[0]}
        block1 = self._find_immediate_win(board_list, PLAYER)
        if block1: print(f"[AI] Blocking immediate loss at {block1}"); return {"x": block1[1], "y": block1[0]}

        # 2. Minimax Search
        print(f"[AI] Starting search: Depth={self.depth}, Workers={self.max_workers}, Size={SIZE}x{SIZE}")
        moves = self._generate_moves(board_list, AI)
        if not moves: return self._fallback_move(board_list) # Handle no moves

        # Heuristic move ordering
        scored_moves = self._score_initial_moves(board_list, moves, AI)
        sorted_moves = [mv for mv, _ in scored_moves] # Keep only moves

        if not sorted_moves: return self._fallback_move(board_list) # Handle no valid moves after scoring

        print(f"[AI] Candidates: {len(sorted_moves)}. Top: {sorted_moves[0]} (Score: {scored_moves[0][1]:.1f})")

        # --- Multiprocessing ---
        init_hash = self._hash_board(board_list)
        futures = {} # future -> move mapping

        # Limit evaluated moves for performance, especially at higher depths
        # More aggressive pruning for deeper searches
        max_moves_to_eval = min(len(sorted_moves), 25 + (5 - min(self.depth, 5)) * 5)
        moves_to_evaluate = sorted_moves[:max_moves_to_eval]
        print(f"[AI] Evaluating top {len(moves_to_evaluate)} moves...")

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for r, c in moves_to_evaluate:
                b2_list = [row[:] for row in board_list]
                # Assume move is valid based on generation/scoring phase
                b2_list[r][c] = AI
                h2 = self._update_hash(init_hash, r, c, AI)
                fut = pool.submit(self._minimax_memo_process_wrapper, # Wrapper for instance method
                                  b2_list, self.depth - 1, False, -math.inf, math.inf, h2)
                futures[fut] = (r, c)

            # Collect results
            results_list = []
            for fut in futures:
                move = futures[fut]
                try:
                    # Timeout can be added here if needed: fut.result(timeout=...)
                    sc = fut.result()
                    print(f"[AI]   Eval {move} -> Score: {sc:.1f}")
                    results_list.append({'move': move, 'score': sc})
                except Exception as e:
                    print(f"[AI] Error evaluating move {move}: {e}")
                    # Assign a very bad score if evaluation fails?
                    results_list.append({'move': move, 'score': -math.inf - 1}) # Mark as failed

        if not results_list:
             print("[AI] WARN: No results collected from workers. Using best initial move.")
             best_move = sorted_moves[0]
        else:
            # Find the best move among successful evaluations
            results_list.sort(key=lambda x: x['score'], reverse=True)
            best_result = results_list[0]
            # Check if the best score is actually viable (not the error score)
            if best_result['score'] > -math.inf - 1:
                 best_move = best_result['move']
                 best_score = best_result['score']
            else:
                 # All evaluations might have failed
                 print("[AI] WARN: All evaluations failed or returned error scores. Using best initial move.")
                 best_move = sorted_moves[0] # Fallback to best heuristic move
                 best_score = scored_moves[0][1] # Use its heuristic score


        # Final fallback if something went critically wrong
        if best_move is None:
            print("[AI] CRITICAL WARN: No best move determined. Using fallback.")
            best_move_coords = self._fallback_move(board_list)
            # Check if fallback returned an error structure
            if "error" in best_move_coords: return best_move_coords
            best_move = (best_move_coords['y'], best_move_coords['x']) # Convert back to (r, c)

        print(f"[AI] Selected move: {best_move} Score: {best_score:.1f} Time: {(time.time()-start_time):.2f}s")
        return {"x": best_move[1], "y": best_move[0]} # Return as (x, y) dict

    def _score_initial_moves(self, board_list, moves, player):
        """Evaluate potential moves heuristically for ordering."""
        scored = []
        temp_board_arr = np.array(board_list, dtype=np.int8)
        opponent = PLAYER if player == AI else AI
        for r, c in moves:
             if 0 <= r < SIZE and 0 <= c < SIZE and temp_board_arr[r, c] == EMPTY:
                  temp_board_arr[r, c] = player
                  # Evaluate score from AI's perspective regardless of who's move it is
                  score = self._evaluate_board(temp_board_arr, AI)
                  # Add bonus for moves near center on larger boards? Maybe not needed if eval handles it.
                  # Add bonus for forcing moves (creating immediate threats)?
                  temp_board_arr[r, c] = EMPTY
                  scored.append(((r, c), score))
        scored.sort(key=lambda x: x[1], reverse=True) # AI wants highest score first
        return scored

    def _fallback_move(self, board_list):
        """Find a fallback move if normal logic fails."""
        print("[AI] Executing fallback move logic.")
        # Try center first
        center_r, center_c = SIZE // 2, SIZE // 2
        if board_list[center_r][center_c] == EMPTY:
             print("[AI] Fallback: Center")
             return {"x": center_c, "y": center_r}
        # Try first available empty cell
        for r in range(SIZE):
            for c in range(SIZE):
                if board_list[r][c] == EMPTY:
                     print(f"[AI] Fallback: First empty cell at ({r}, {c})")
                     return {"x": c, "y": r}
        # Should not happen if game isn't over
        print("[AI] Fallback ERROR: No empty cells found!")
        return {"error": "Board is full or AI error during fallback"}

    def _find_immediate_win(self, board_list, player):
        """Checks if 'player' can win in one move."""
        potential_moves = self._generate_moves(board_list, player)
        temp_board_arr = np.array(board_list, dtype=np.int8)
        player_np = np.int8(player)
        for r, c in potential_moves:
            if 0 <= r < SIZE and 0 <= c < SIZE and temp_board_arr[r, c] == EMPTY:
                temp_board_arr[r, c] = player_np
                won = _check_win_numba(temp_board_arr, player_np, c, r, SIZE)
                temp_board_arr[r, c] = EMPTY
                if won: return (r, c)
        return None

    # Wrapper needed because instance methods aren't directly picklable for ProcessPoolExecutor
    def _minimax_memo_process_wrapper(self, board_list, depth, is_max, alpha, beta, h):
         # Each process gets its own transposition table copy (passed implicitly via self)
         # Or maybe pass table explicitly if instance state isn't reliable across processes?
         # Let's try using the instance table for now, assuming ProcessPoolExecutor handles it.
         # If issues arise, create table={} here and pass it recursively.
         return self._minimax_memo(board_list, depth, is_max, alpha, beta, h, {}) # Pass empty table

    def _minimax_memo(self, board_list, depth, is_max, alpha, beta, h, table):
        """Minimax with Alpha-Beta and Transposition Table."""
        key = (h, depth, is_max)
        if key in table: return table[key]

        board_arr = np.array(board_list, dtype=np.int8) # Needed for eval/check
        current_player = AI if is_max else PLAYER
        opponent = PLAYER if is_max else AI

        # Check for terminal state (Win/Loss detected by opponent's last move simulation)
        # Instead of full board check, use evaluate score - it includes win checks
        current_eval_score = self._evaluate_board(board_arr, AI) # Evaluate from AI perspective
        if abs(current_eval_score) >= SCORE_FIVE * 5: # Use threshold based on eval score
             # Score needs to reflect depth: win sooner = better, lose later = better
             # High positive score means AI won, High negative means Player won
             # Return score relative to the *maximizing* player (AI)
             score = current_eval_score * (depth + 1)
             table[key] = score
             return score

        if depth == 0:
            table[key] = current_eval_score # Return static evaluation at leaf
            return current_eval_score

        moves = self._generate_moves(board_list, current_player)
        if not moves: table[key] = current_eval_score; return current_eval_score # No moves -> draw state? return static eval

        # Move ordering inside minimax recursion
        ordered_moves = self._order_minimax_moves(board_list, moves, current_player, is_max)

        # Alpha-Beta
        best_val = -math.inf if is_max else math.inf
        for r, c in ordered_moves:
            # Ensure move is valid *again* (paranoid check)
            if board_list[r][c] == EMPTY:
                 board_list[r][c] = current_player
                 h2 = self._update_hash(h, r, c, current_player)
                 val = self._minimax_memo(board_list, depth - 1, not is_max, alpha, beta, h2, table)
                 board_list[r][c] = EMPTY # Undo move

                 if is_max:
                     best_val = max(best_val, val)
                     alpha = max(alpha, best_val)
                 else:
                     best_val = min(best_val, val)
                     beta = min(beta, best_val)

                 if beta <= alpha: break # Prune

        table[key] = best_val
        return best_val

    def _order_minimax_moves(self, board_list, moves, player, is_max):
        """Score moves for ordering within minimax recursion."""
        scored = []
        temp_board_arr = np.array(board_list, dtype=np.int8)
        for r_mm, c_mm in moves:
             if temp_board_arr[r_mm, c_mm] == EMPTY:
                  temp_board_arr[r_mm, c_mm] = player
                  # Use static evaluation for ordering
                  sc_mm = self._evaluate_board(temp_board_arr, AI) # Always from AI perspective
                  temp_board_arr[r_mm, c_mm] = EMPTY
                  scored.append(((r_mm, c_mm), sc_mm))
        # Max player explores high scores first, Min player explores low scores first
        scored.sort(key=lambda x: x[1], reverse=is_max)
        # Limit branching factor? Maybe only take top N moves?
        # max_branch = 15 + (3 - min(self.depth, 3)) * 5 # Fewer branches deeper?
        # return [mv for mv, _ in scored[:max_branch]]
        return [mv for mv, _ in scored]


    def _generate_moves(self, board_list, player_to_move):
        """Generates potential moves near existing pieces."""
        moves = set()
        has_piece = False
        radius = 1 # Check immediate neighbors

        for r in range(SIZE):
            for c in range(SIZE):
                if board_list[r][c] != EMPTY:
                    has_piece = True
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < SIZE and 0 <= nc < SIZE and board_list[nr][nc] == EMPTY:
                                moves.add((nr, nc))

        if not has_piece: return [(SIZE // 2, SIZE // 2)] # Play center if empty
        if not moves: # If no neighbors (near full board), find any empty
            empty_spots = []
            for r in range(SIZE):
                for c in range(SIZE):
                    if board_list[r][c] == EMPTY: empty_spots.append((r,c))
            return empty_spots
        return list(moves)

    def _evaluate_board(self, board_arr: np.ndarray, who: np.int8) -> float:
        """Evaluates board state from AI's perspective using Numba."""
        my_score = _calculate_player_score_numba(
            board_arr, AI, PLAYER, SIZE, *self.score_constants
        )
        oppo_score = _calculate_player_score_numba(
            board_arr, PLAYER, AI, SIZE, *self.score_constants
        )

        # Check for win conditions detected by scoring
        # Multiply by large factor to dominate simple scores
        if my_score >= SCORE_FIVE: return SCORE_FIVE * 10
        if oppo_score >= SCORE_FIVE: return -SCORE_FIVE * 10

        # Return AI score minus weighted opponent score
        return float(my_score - oppo_score * 1.1) # Weight blocking slightly


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json or {}
    if 'board' not in data or 'depth' not in data:
        return jsonify({"error":"Missing 'board' or 'depth' parameter"}), 400

    board_list = data['board']
    depth = int(data['depth'])
    workers = int(data.get('threads', os.cpu_count() or 1)) # Still use 'threads' key for backward compatibility

    # --- Input Validation ---
    # MODIFIED: Validate against new ranges (1-20 depth, 1-200 workers)
    if not (1 <= depth <= 20):
        return jsonify({"error": f"搜索深度 (depth) 必须在 1 到 20 之间"}), 400
    if not (1 <= workers <= 200):
         return jsonify({"error": f"并行工作单元 (threads) 必须在 1 到 200 之间"}), 400

    # Print warning for potentially very long calculations
    if depth > 5:
         print(f"WARN: Requested depth {depth} on {SIZE}x{SIZE} board may take a very long time!")

    # Validate board structure and values (SIZE=30)
    try:
        if not (isinstance(board_list, list) and len(board_list) == SIZE and
                all(isinstance(row, list) and len(row) == SIZE for row in board_list) and
                all(cell in [int(EMPTY), int(PLAYER), int(AI)] for row in board_list for cell in row)):
            raise ValueError(f"Invalid board structure or cell values (expected {SIZE}x{SIZE})")
    except ValueError as e:
         return jsonify({"error": f"Board format is invalid: {e}"}), 400

    # --- Create AI instance and get move ---
    print(f"Received request: depth={depth}, workers={workers} (Numba Enabled, Board Size: {SIZE}x{SIZE})")
    ai = GomokuAI(board_list, depth, workers)
    try:
        # Consider adding a timeout mechanism here for very long searches
        move = ai.find_best_move()

        if move is None or "error" in move:
             # AI class should handle fallbacks, but catch potential None return
             print("ERROR: AI find_best_move returned None or an error structure.")
             err_msg = move.get("error", "AI failed to determine a move") if isinstance(move, dict) else "AI failed"
             # Try one last time to find *any* empty spot as absolute fallback
             final_fallback = None
             for r in range(SIZE):
                 for c in range(SIZE):
                     if board_list[r][c] == int(EMPTY):
                         final_fallback = {"x": c, "y": r}
                         break
                 if final_fallback: break
             if final_fallback:
                  print("WARN: Using absolute fallback move.")
                  return jsonify({"move": final_fallback, "warning": f"{err_msg}, used absolute fallback"})
             else: # Truly no empty spots?
                  return jsonify({"error": "AI failed and no empty cells found (Draw?)"}), 500

        print(f"Sending move: {move}")
        return jsonify({"move": move})

    except Exception as e:
        print("="*20 + " AI Calculation Error " + "="*20)
        print(f"Request Data: depth={depth}, workers={workers}, size={SIZE}")
        traceback.print_exc()
        print("="*50)
        # Provide a more generic error message to the frontend
        return jsonify({"error": f"AI internal calculation error. Check server logs."}), 500


# --- Startup ---
if __name__=='__main__':
    # freeze_support() should be in main.py if bundling

    port = 5000
    print(f"=== Gomoku AI Server (Multiprocessing + Numba) starting on port {port} ===")
    print(f"Board Size: {SIZE}x{SIZE}")
    print(f"Detected CPU cores: {os.cpu_count() or 'Unknown'}")
    print(f"Numba available: {'Yes' if numba else 'No'}")

    # Pre-compile Numba functions (optional, uses SIZE=30)
    try:
        print("Pre-compiling Numba functions (this might take a moment)...")
        dummy_board_arr = np.zeros((SIZE, SIZE), dtype=np.int8)
        dummy_line = np.zeros(SIZE, dtype=np.int8)
        dummy_scores = (
            SCORE_FIVE, SCORE_LIVE_FOUR, SCORE_RUSH_FOUR, SCORE_LIVE_THREE,
            SCORE_SLEEP_THREE, SCORE_LIVE_TWO, SCORE_SLEEP_TWO
        )
        _check_win_numba(dummy_board_arr, PLAYER, SIZE//2, SIZE//2, SIZE)
        _score_pattern_numba(3, 2, *dummy_scores)
        _evaluate_line_numba(dummy_line, PLAYER, AI, *dummy_scores)
        _calculate_player_score_numba(dummy_board_arr, PLAYER, AI, SIZE, *dummy_scores)
        print("Numba pre-compilation finished.")
    except Exception as e:
        print(f"Numba pre-compilation failed (will compile on first use): {e}")

    # Run Flask server
    app.run(host='127.0.0.1', port=port, debug=False, threaded=False, use_reloader=False)