# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
import copy

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
    "CENTER_BONUS":  1
}

# --- 公共工具函数 ---
def is_valid(board, x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    """判断 player 在 (x,y) 下子后是否五连"""
    if not (0 <= x < SIZE and 0 <= y < SIZE and board[y][x] == player):
        return False
    directions = [(1,0),(0,1),(1,1),(1,-1)]
    for dx,dy in directions:
        cnt = 1
        # 正方向
        for i in range(1,5):
            nx,ny = x+i*dx, y+i*dy
            if not is_valid(board,nx,ny) or board[ny][nx] != player:
                break
            cnt+=1
        # 反方向
        for i in range(1,5):
            nx,ny = x-i*dx, y-i*dy
            if not is_valid(board,nx,ny) or board[ny][nx] != player:
                break
            cnt+=1
        if cnt>=5:
            return True
    return False

# --- AI 核心 ---
class GomokuAI:
    # Zobrist 哈希表，全局一次性初始化
    zobrist_table = [[[random.getrandbits(64) for _ in range(3)]
                        for _ in range(SIZE)]
                       for _ in range(SIZE)]
    empty_board_hash = 0

    def __init__(self, board, depth, max_workers=None):
        # board: 二维列表，depth: 搜索深度(1~12)，max_workers: 线程数(1~12)
        self.initial_board = [row[:] for row in board]
        self.depth = depth
        self.max_workers = max_workers

    def _hash_board(self, board):
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                p = board[r][c]
                if p != EMPTY:
                    h ^= self.zobrist_table[r][c][p]
        return h

    def _update_hash(self, h, r, c, player):
        # 落子或撤销都只需一次 XOR
        return h ^ self.zobrist_table[r][c][player]

    def find_best_move(self):
        """对外接口：返回字典 {'x':col,'y':row}"""
        start_time = time.time()
        best_score = -math.inf
        best_move  = None
        board = [row[:] for row in self.initial_board]

        # 1. 先一步制胜？
        win1 = self._find_immediate_win(board, AI)
        if win1:
            return {"x": win1[1], "y": win1[0]}

        # 2. 玩家一步制胜？去阻挡
        block1 = self._find_immediate_win(board, PLAYER)
        if block1:
            return {"x": block1[1], "y": block1[0]}

        # 3. Minimax+αβ+置换表+多线程
        print(f"[AI] 搜索深度 depth={self.depth} 线程数 max_workers={self.max_workers}")
        # 候选走法
        moves = self._generate_moves(board, AI)
        if not moves:
            # 如果真的没地方下，就随便找一个空
            for rr in range(SIZE):
                for cc in range(SIZE):
                    if board[rr][cc]==EMPTY:
                        return {"x":cc,"y":rr}
            return {"x":0,"y":0}

        # 初步启发式排序
        scored = []
        for r,c in moves:
            board[r][c] = AI
            sc = self._evaluate_board(board, AI)
            board[r][c] = EMPTY
            scored.append(((r,c),sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_moves = [mv for mv,_ in scored]

        # 置换表
        trans_table = {}
        init_hash = self._hash_board(board)

        # 确定线程数
        if isinstance(self.max_workers,int):
            mw = self.max_workers
        else:
            mw = os.cpu_count() or 1
        max_workers = max(1, min(mw, 100))
        print(f"[AI] ThreadPoolExecutor max_workers={max_workers}")
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for r,c in sorted_moves:
                b2 = [row[:] for row in board]
                b2[r][c] = AI
                h2 = self._update_hash(init_hash, r, c, AI)
                fut = pool.submit(self._minimax_memo,
                                  b2, self.depth-1, False,
                                  -math.inf, math.inf,
                                  h2, trans_table)
                futures.append((fut,(r,c)))

            for fut,(r,c) in futures:
                try:
                    sc = fut.result()
                    if sc > best_score:
                        best_score = sc
                        best_move  = (r,c)
                except Exception as e:
                    # 某个线程失败时忽略
                    print("[AI] 子线程异常", e)

        if best_move is None:
            best_move = sorted_moves[0]

        print(f"[AI] 选定走法 {best_move} 分数={best_score:.1f} 耗时 {(time.time()-start_time):.2f}s")
        return {"x":best_move[1], "y":best_move[0]}

    def _find_immediate_win(self, board, player):
        """找到 player 一步必胜的位置"""
        for r,c in self._generate_moves(board, player):
            if board[r][c]!=EMPTY: continue
            board[r][c]=player
            if check_win(board, player, c, r):
                board[r][c]=EMPTY
                return (r,c)
            board[r][c]=EMPTY
        return None

    def _minimax_memo(self, board, depth, is_max, alpha, beta, h, table):
        """带置换表的 Minimax (递归)"""
        key = (h, depth, is_max)
        if key in table:
            return table[key]
        # 终止条件：深度0 or 无路可走
        if depth==0:
            sc = self._evaluate_board(board, AI)
            table[key]=sc
            return sc

        player = AI if is_max else PLAYER
        moves = self._generate_moves(board, player)
        if not moves:
            sc = self._evaluate_board(board, AI)
            table[key]=sc
            return sc

        if is_max:
            val = -math.inf
            for r,c in moves:
                board[r][c] = AI
                h2 = self._update_hash(h, r, c, AI)
                tmp = self._minimax_memo(board, depth-1, False, alpha, beta, h2, table)
                board[r][c] = EMPTY
                val = max(val, tmp)
                alpha = max(alpha, tmp)
                if beta <= alpha:
                    break
            table[key]=val
            return val
        else:
            val = math.inf
            for r,c in moves:
                board[r][c] = PLAYER
                h2 = self._update_hash(h, r, c, PLAYER)
                tmp = self._minimax_memo(board, depth-1, True, alpha, beta, h2, table)
                board[r][c] = EMPTY
                val = min(val, tmp)
                beta = min(beta, tmp)
                if beta <= alpha:
                    break
            table[key]=val
            return val

    def _generate_moves(self, board, player_to_move):
        """启发式生成候选走法：空位邻近已有棋子的点"""
        moves = set()
        has_piece = False
        radius = 1
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c]!=EMPTY:
                    has_piece = True
                    for dr in range(-radius, radius+1):
                        for dc in range(-radius, radius+1):
                            if dr==0 and dc==0: continue
                            nr, nc = r+dr, c+dc
                            if is_valid(board,nc,nr) and board[nr][nc]==EMPTY:
                                moves.add((nr,nc))
        # 若空棋盘
        if not has_piece:
            mid = SIZE//2
            return [(mid,mid)]
        return list(moves) if moves else [(r,c) for r in range(SIZE) for c in range(SIZE) if board[r][c]==EMPTY]

    def _evaluate_board(self, board, who):
        """整体评估：AI 分数 减去 玩家分数 * 1.1"""
        my    = self._calc_score(board, AI)
        oppo  = self._calc_score(board, PLAYER)
        if my   >= SCORE["FIVE"]: return SCORE["FIVE"]*10
        if oppo >= SCORE["FIVE"]: return -SCORE["FIVE"]*10
        return my - oppo*1.1

    def _calc_score(self, board, player):
        """计算指定玩家在当前棋盘上的总得分"""
        total_score = 0
        evaluated_lines = set() # 用于避免重复计算同一条线的不同部分

        lines = self._get_all_lines(board)
        for line in lines:
             line_tuple = tuple(line) # Convert list to tuple for hashing
             if line_tuple not in evaluated_lines:
                 total_score += self._evaluate_line_fast(line, player)
                 evaluated_lines.add(line_tuple)

        return total_score

    def _get_all_lines(self, board):
        """获取棋盘上所有长度至少为5的行、列和对角线"""
        lines = []
        # Rows
        for r in range(SIZE):
            lines.append(board[r])
        # Columns
        for c in range(SIZE):
            lines.append([board[r][c] for r in range(SIZE)])
        # Diagonals (top-left to bottom-right)
        for i in range(-(SIZE - 5), SIZE - 4): # Adjusted range for length >= 5
            line = []
            for j in range(SIZE):
                r, c = j, j + i
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(board[r][c])
            if len(line) >= 5: lines.append(line)
        # Diagonals (top-right to bottom-left)
        for i in range(4, 2 * SIZE - 5): # Adjusted range for length >= 5
            line = []
            for j in range(SIZE):
                r, c = j, i - j
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(board[r][c])
            if len(line) >= 5: lines.append(line)
        return lines

    def _evaluate_line_fast(self, line, player):
        """
        更快速地评估一条线的分数（仅使用模式匹配思想，单次遍历）。
        :param line: list of cells (0, 1, or 2)
        :param player: the player for whom to calculate the score (1 or 2)
        :return: score for this line for the given player
        """
        score = 0
        n = len(line)
        opponent = PLAYER if player == AI else AI
        p_char = str(player)
        o_char = str(opponent)
        e_char = str(EMPTY)

        # 将列表转换为字符串以便查找，这可能仍然是瓶颈，但比之前的双重评估好
        # 可以考虑纯粹的列表遍历匹配，但实现更复杂
        line_str = "".join(map(str, line))

        # --- 核心棋型评分 ---
        # 五连 (单独处理，最高优先级)
        if p_char * 5 in line_str:
            return SCORE["FIVE"] # 直接返回最高分

        # 活四: E PPPP E
        if e_char + p_char*4 + e_char in line_str:
            score += SCORE["LIVE_FOUR"]

        # 冲四: O PPPP E or E PPPP O or PPP EP or PP E PP or P E PPP
        # (注意顺序和边界 O)
        rush4_patterns = [
            o_char + p_char*4 + e_char,
            e_char + p_char*4 + o_char,
            p_char*3 + e_char + p_char, # P P P E P
            p_char + e_char + p_char*3, # P E P P P
            p_char*2 + e_char + p_char*2, # P P E P P
        ]
        # 需要考虑边界情况，例如 " PPPPE" 或 "EPPPP "
        # 简单的 in 检查可能不够，需要更复杂的检查或确保 line 两端有足够空间或特殊标记
        # 为了简化，我们暂时依赖 SCORE["RUSH_FOUR"] 的值小于 LIVE_FOUR
        found_rush4 = False
        for pattern in rush4_patterns:
             if pattern in line_str:
                 score += SCORE["RUSH_FOUR"]
                 found_rush4 = True
                 # 找到一个冲四就不再找其他冲四模式，避免重复加分（一个活四可能包含冲四模式）
                 break
        # 确保冲四分数不会超过活四
        if found_rush4 and score >= SCORE["LIVE_FOUR"]:
             score = SCORE["LIVE_FOUR"] -1 # 或者取 RUSH_FOUR 和 LIVE_FOUR 的较小值

        # 活三: E PPP E E or E E PPP E
        # Also: E P EP PE
        live3_patterns = [
             e_char + p_char*3 + e_char, # E P P P E
             e_char + p_char + e_char + p_char*2 + e_char, # E P E P P E
             e_char + p_char*2 + e_char + p_char + e_char, # E P P E P E
        ]
        found_live3 = False
        for pattern in live3_patterns:
            # 使用 find 而不是 in 来统计次数，避免重叠模式重复计数
            start_index = 0
            while True:
                 index = line_str.find(pattern, start_index)
                 if index == -1: break
                 # 检查边界是否是 E 或 O (更严格的活三定义)
                 left_ok = (index == 0 or line_str[index-1] == e_char)
                 right_ok = (index + len(pattern) == n or line_str[index + len(pattern)] == e_char)

                 # 附加条件: EPPP E 两侧必须有空位才能发展成活四
                 is_true_live_three = False
                 if pattern == e_char + p_char*3 + e_char:
                     # 需要检查 E P P P E 两边的空位 E E P P P E E
                     left_further_ok = (index > 0 and line_str[index-1] == e_char)
                     right_further_ok = (index + len(pattern) < n and line_str[index+len(pattern)] == e_char)
                     if left_further_ok and right_further_ok:
                         is_true_live_three = True
                 elif pattern == e_char + p_char + e_char + p_char*2 + e_char: # E P E P P E
                      is_true_live_three = True # 通常认为是活三
                 elif pattern == e_char + p_char*2 + e_char + p_char + e_char: # E P P E P E
                      is_true_live_three = True # 通常认为是活三

                 if is_true_live_three:
                     score += SCORE["LIVE_THREE"]
                     found_live3 = True
                 else: # 如果不是真活三 (例如 O E P P P E E)，算作眠三
                     score += SCORE["SLEEP_THREE"]

                 start_index = index + 1 # 继续查找下一个
            if found_live3: break # 找到一个活三即可

        # 眠三: O PPP E or E PPP O or P P EP O or P EP P O or ...
        # 模式非常多，这里简化处理：如果不是活三，但存在三个子，认为是眠三
        # 这个逻辑比较粗糙，更好的方法是列举所有眠三模式
        # 或者在上面的活三检查中，不满足条件的都归为眠三
        # (上面活三检查已部分处理 O E P P P E E)
        # 补充一些常见的眠三模式
        sleep3_patterns = [
             o_char + p_char*3 + e_char, # O P P P E
             e_char + p_char*3 + o_char, # E P P P O
             o_char + p_char + e_char + p_char*2 + e_char, # O P E P P E (被 O 挡住一边)
             e_char + p_char*2 + e_char + p_char + o_char, # E P P E P O (被 O 挡住一边)
             p_char + e_char + p_char + e_char + p_char, # P E P E P (可能构成眠三或更复杂棋型)
             p_char*2 + e_char*2 + p_char, # P P E E P
             p_char + e_char*2 + p_char*2, # P E E P P
        ]
        # 避免重复计数，如果已计为活三，则不计眠三
        if not found_live3:
            for pattern in sleep3_patterns:
                 if pattern in line_str:
                     # 检查是否已被计为冲四（冲四 > 眠三）
                     if not found_rush4 or SCORE["RUSH_FOUR"] < SCORE["SLEEP_THREE"]: # 理论上不会发生
                        score += SCORE["SLEEP_THREE"]
                        # 找到一个即可，避免例如 O P P P E O 计两次
                        break

        # 活二: E E PP E E or E P EP E E or E P E EP E
        live2_patterns = [
            e_char*2 + p_char*2 + e_char*2, # E E P P E E
            e_char + p_char + e_char + p_char + e_char*2, # E P E P E E
            e_char*2 + p_char + e_char + p_char + e_char, # E E P E P E
            e_char + p_char + e_char*2 + p_char + e_char, # E P E E P E
        ]
        found_live2 = False
        for pattern in live2_patterns:
             if pattern in line_str:
                 score += SCORE["LIVE_TWO"]
                 found_live2 = True
                 # 找到一个即可
                 break

        # 眠二: O PP E E or E E PP O or O P EP E or E P EP O ...
        # 简化：如果不是活二，但存在两个子，认为是眠二
        sleep2_patterns = [
            o_char + p_char*2 + e_char*2, # O P P E E
            e_char*2 + p_char*2 + o_char, # E E P P O
            o_char + p_char + e_char + p_char + e_char, # O P E P E
            e_char + p_char + e_char + p_char + o_char, # E P E P O
            e_char + p_char + e_char*2 + p_char + o_char, # E P E E P O
             # EEEPEEE
             e_char*3 + p_char + e_char*3,
             # EEEP EE
             e_char*3 + p_char*2 + e_char*2,
             e_char*2 + p_char*2 + e_char*3,
        ]
        if not found_live2:
             for pattern in sleep2_patterns:
                 if pattern in line_str:
                     # 确保不与更高级棋型冲突
                     if not (found_rush4 or found_live3): # (眠三 > 眠二)
                        score += SCORE["SLEEP_TWO"]
                        break # 找到一个即可

        # 对于单个棋子 (活一/眠一)，其价值相对较低，主要体现在构成二连或三连的潜力中
        # 可以简化评估，不单独计算活一/眠一，或者给予非常低的分数
        # score += line_str.count(e_char + p_char + e_char) * SCORE["LIVE_ONE"] # E P E
        # score += (line_str.count(o_char + p_char + e_char) + line_str.count(e_char + p_char + o_char)) * SCORE["SLEEP_ONE"] # O P E or E P O

        return score

# --- Flask 路由 ---  
@app.route('/', methods=['GET'])
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json or {}
    if 'board' not in data or 'depth' not in data:
        return jsonify({"error":"缺少 board 或 depth"}),400

    board = data['board']
    depth = int(data['depth'])
    threads = int(data.get('threads', os.cpu_count() or 1))

    # 验证
    if depth<1 or depth>12:
        return jsonify({"error":"depth 必须在1~12之间"}),400
    if threads<1 or threads>100:
        return jsonify({"error":"threads 必须在1~100之间"}),400
    if not (isinstance(board,list) and len(board)==SIZE and all(isinstance(r,list) and len(r)==SIZE for r in board)):
        return jsonify({"error":"board 格式错误"}),400

    ai = GomokuAI(board, depth, threads)
    try:
        mv = ai.find_best_move()
        return jsonify({"move":mv})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error":"AI内部错误"}),500

# --- 启动 ---
if __name__=='__main__':
    port = 5000
    print(f"=== Gomoku AI Server 启动，端口 {port} ===")
    app.run(host='127.0.0.1', port=port, debug=False)
