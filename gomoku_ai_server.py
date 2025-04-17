# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# 导入 ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor # <--- 修改：使用 ProcessPoolExecutor
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
    "CENTER_BONUS":  1 # 这个好像没用到，可以考虑加到评估里或移除
}

# Zobrist 哈希表，全局一次性初始化 (保持不变)
zobrist_table = [[[random.getrandbits(64) for _ in range(3)]
                    for _ in range(SIZE)]
                   for _ in range(SIZE)]
empty_board_hash = 0 # 这个好像也没用到

# --- 公共工具函数 (保持不变) ---
def is_valid(board, x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    """判断 player 在 (x,y) 下子后是否五连"""
    if not (0 <= x < SIZE and 0 <= y < SIZE): return False # 先检查坐标有效性
    if board[y][x] != player: return False # 确保该点是刚下的子

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

# 将 _minimax_memo 函数移出类，或者作为静态方法，以便于多进程调用
# 这里我们选择保持它在类内部，因为 ProcessPoolExecutor 通常能处理实例方法
# 但需要注意，传递给子进程的是序列化的对象或参数

class GomokuAI:
    # Zobrist table 是类变量，会被所有实例共享，这没问题
    # 但在多进程中，每个进程会有自己的内存空间，实际上是复制了这份表

    def __init__(self, board, depth, max_workers=None):
        self.initial_board = [row[:] for row in board]
        self.depth = depth
        self.max_workers = max_workers if max_workers is not None else os.cpu_count() or 1
        # 限制最大工作单元数，防止过多进程导致开销过大
        self.max_workers = max(1, min(self.max_workers, os.cpu_count() or 4)) # 例如最多不超过 CPU 核心数

    def _hash_board(self, board):
        # Zobrist 哈希计算 (保持不变)
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                p = board[r][c]
                if p != EMPTY:
                    h ^= zobrist_table[r][c][p]
        return h

    def _update_hash(self, h, r, c, player):
        # Zobrist 哈希更新 (保持不变)
        return h ^ zobrist_table[r][c][player]

    def find_best_move(self):
        start_time = time.time()
        best_score = -math.inf
        best_move  = None
        board = [row[:] for row in self.initial_board]

        # 1. 先一步制胜？(保持不变)
        win1 = self._find_immediate_win(board, AI)
        if win1:
            print(f"[AI] 立即获胜于 {win1}")
            return {"x": win1[1], "y": win1[0]}

        # 2. 玩家一步制胜？去阻挡 (保持不变)
        block1 = self._find_immediate_win(board, PLAYER)
        if block1:
            print(f"[AI] 立即阻止对手获胜于 {block1}")
            return {"x": block1[1], "y": block1[0]}

        # 3. Minimax+αβ+置换表+多进程
        print(f"[AI] 搜索深度 depth={self.depth} 并行工作单元数 max_workers={self.max_workers} (使用多进程)")
        moves = self._generate_moves(board, AI)
        if not moves:
            print("[AI] 没有可选的移动了?")
            for rr in range(SIZE):
                for cc in range(SIZE):
                    if board[rr][cc]==EMPTY: return {"x":cc,"y":rr}
            return {"x":0,"y":0} # 应该不会发生

        # 启发式排序 (保持不变)
        scored = []
        for r,c in moves:
            board[r][c] = AI
            sc = self._evaluate_board(board, AI) # 使用优化后的评估函数
            board[r][c] = EMPTY
            scored.append(((r,c),sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_moves = [mv for mv,_ in scored]
        print(f"[AI] 候选走法数量: {len(sorted_moves)}. 最佳初步评估走法: {sorted_moves[0]} 分数: {scored[0][1]:.1f}" if sorted_moves else "[AI] 无候选走法")


        # 置换表 - 每个进程将使用自己的独立表
        init_hash = self._hash_board(board)

        futures = []
        results = {} # 存储 future -> move 的映射

        # 使用 ProcessPoolExecutor
        # 注意：max_workers 最好不要超过 CPU 核心数太多
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for r,c in sorted_moves:
                # 为每个任务创建棋盘副本
                b2 = [row[:] for row in board]
                b2[r][c] = AI
                h2 = self._update_hash(init_hash, r, c, AI)

                # 提交任务给进程池
                # _minimax_memo 方法及其参数需要能被 pickle
                # 传递独立的置换表 {} 给每个进程
                fut = pool.submit(self._minimax_process_wrapper, # 使用包装器传递必要数据
                                  b2, self.depth-1, False,
                                  -math.inf, math.inf,
                                  h2)
                futures.append(fut)
                results[fut] = (r,c) # 关联 future 和 move

            # 收集结果
            # 按提交顺序或完成顺序处理都可以，这里按完成顺序
            # from concurrent.futures import as_completed
            # for fut in as_completed(futures):
            #     move = results[fut]
            #     try:
            #         sc = fut.result()
            #         print(f"[AI]   评估走法 {move} -> 分数: {sc:.1f}")
            #         if sc > best_score:
            #             best_score = sc
            #             best_move  = move
            #     except Exception as e:
            #         print(f"[AI] 子进程评估走法 {move} 时发生异常: {e}")
            #         # 可以选择给这个走法一个最低分，或者忽略它
            #         if best_move is None: # 如果还没有最佳走法，至少选第一个
            #              best_move = sorted_moves[0]

            # 按提交顺序获取结果（更简单）
            for i, fut in enumerate(futures):
                 move = results[fut]
                 try:
                     sc = fut.result() # 等待结果
                     print(f"[AI]   评估走法 {move} -> 分数: {sc:.1f}")
                     if sc > best_score:
                         best_score = sc
                         best_move = move
                 except Exception as e:
                     print(f"[AI] 子进程评估走法 {move} 时发生异常: {e}")
                     import traceback
                     traceback.print_exc() # 打印详细错误
                     # 异常处理：可以给一个极低分，或者依赖于启发式排序的初始结果
                     if best_move is None and i == 0: # 如果是第一个评估就出错，且还没有最佳选择
                         best_move = move # 至少选第一个

        # 如果所有进程都失败了，或者没有找到更好的分数，选择启发式排序最好的那个
        if best_move is None:
            print("[AI] 所有子进程评估失败或无更优解，选择初步评估最佳走法")
            best_move = sorted_moves[0]

        print(f"[AI] 选定走法 {best_move} 分数={best_score:.1f} 耗时 {(time.time()-start_time):.2f}s")
        return {"x":best_move[1], "y":best_move[0]}

    # --- 需要一个包装函数来传递给 ProcessPoolExecutor ---
    # 因为类实例方法可能在某些情况下不易 pickle，或者需要传递额外的类/静态数据
    # 这个包装器接收参数，并调用实际的 minimax 方法
    # 注意：这个包装器本身需要在模块顶层或者可以通过 pickle 找到
    # 或者，确保 GomokuAI 实例和 _minimax_memo 方法是可 pickle 的
    # 我们尝试直接调用 _minimax_memo，如果不行再用包装器

    # 包装函数示例 (如果直接调用方法失败时使用)
    # def _minimax_process_wrapper(self, board_copy, depth, is_max, alpha, beta, current_hash):
    #     # 在子进程中，需要能够访问评估和生成函数
    #     # 以及常量 SCORE, SIZE 等。
    #     # 这里假设 _minimax_memo 可以访问它们 (因为它仍在类定义内，或者它们是全局的)
    #     # 每个进程创建自己的置换表
    #     transposition_table = {}
    #     return self._minimax_memo(board_copy, depth, is_max, alpha, beta, current_hash, transposition_table)

    # 静态方法版本 (如果实例方法pickle有问题)
    @staticmethod
    def _minimax_process_wrapper(board_copy, depth, is_max, alpha, beta, current_hash, zobrist_table_ref, score_ref, size_ref, player_ref, ai_ref, empty_ref):
        # 需要显式传递所有依赖项，或者重新定义/导入它们
        # 这里只是示例结构
        table = {}
        # 需要一种方法来调用 minimax 逻辑，可能需要重构 minimax 使其不依赖 self
        # 或者创建一个临时的 GomokuAI 实例? (效率低)
        # 最好的方式还是确保 _minimax_memo 可以被 pickle 调用

        # --> 实际上，concurrent.futures 通常能处理实例方法，我们先尝试直接调用
        # 如果遇到 PicklingError，再回来实现这个包装器或者将 minimax 改为静态方法
        pass # 占位

    def _find_immediate_win(self, board, player):
        # (保持不变)
        for r,c in self._generate_moves(board, player): # 只在潜在的落子点检查
            if board[r][c]==EMPTY:
                board[r][c]=player
                # check_win 现在需要 board, player, x, y
                if check_win(board, player, c, r): # 传递 x, y
                    board[r][c]=EMPTY
                    return (r,c)
                board[r][c]=EMPTY
        return None

    def _minimax_memo(self, board, depth, is_max, alpha, beta, h, table):
        # (保持不变，除了接收 table 参数)
        # table 现在是每个进程/线程独立的
        key = (h, depth, is_max)
        if key in table:
            return table[key]

        # 检查是否有立即获胜/失败 (可选但推荐)
        winner = self._check_board_winner(board) # 需要一个快速检查函数
        if winner == AI:
             table[key] = SCORE["FIVE"] * (depth + 1) # 深度越高越快获胜，分数越高
             return table[key]
        if winner == PLAYER:
             table[key] = -SCORE["FIVE"] * (depth + 1)
             return table[key]

        if depth == 0:
            # 评估叶子节点 (使用优化后的评估函数)
            sc = self._evaluate_board(board, AI) # 总是从 AI 的角度评估
            table[key]=sc
            return sc

        player = AI if is_max else PLAYER
        moves = self._generate_moves(board, player)

        if not moves: # 没有可走的路
            sc = self._evaluate_board(board, AI)
            table[key]=sc
            return sc

        # 对子节点进行排序（可选但推荐，改进Alpha-Beta效率）
        # 这里可以在递归中也做启发式排序，但会增加开销
        # sorted_moves = self._sort_moves(board, moves, player) # 需要实现 _sort_moves

        if is_max: # AI (Maximizing player)
            best_val = -math.inf
            for r, c in moves: # or sorted_moves
                board[r][c] = AI
                h2 = self._update_hash(h, r, c, AI)
                val = self._minimax_memo(board, depth-1, False, alpha, beta, h2, table)
                board[r][c] = EMPTY # 撤销
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break # Beta cut-off
            table[key] = best_val
            return best_val
        else: # Player (Minimizing player)
            best_val = math.inf
            for r, c in moves: # or sorted_moves
                board[r][c] = PLAYER
                h2 = self._update_hash(h, r, c, PLAYER)
                val = self._minimax_memo(board, depth-1, True, alpha, beta, h2, table)
                board[r][c] = EMPTY # 撤销
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break # Alpha cut-off
            table[key] = best_val
            return best_val

    def _check_board_winner(self, board):
        """快速检查棋盘上是否已有五连"""
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != EMPTY:
                    # 只需要检查一个方向（例如右、下、右下、右上）避免重复
                    # 这里为了简单，还是用 check_win，但可以优化
                    if check_win(board, board[r][c], c, r):
                        return board[r][c]
        return EMPTY # No winner

    def _generate_moves(self, board, player_to_move):
        # (保持不变)
        moves = set()
        has_piece = False
        # 搜索半径，可以调整，例如 1 或 2
        radius = 1 # 只考虑紧邻棋子的空位
        # 也可以考虑更大的半径，或者基于威胁（如对方活三、冲四点）
        # radius = 2

        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c]!=EMPTY:
                    has_piece = True
                    for dr in range(-radius, radius+1):
                        for dc in range(-radius, radius+1):
                            if dr==0 and dc==0: continue
                            nr, nc = r+dr, c+dc
                            # 确保在棋盘内且为空位
                            if is_valid(board, nc, nr) and board[nr][nc]==EMPTY:
                                moves.add((nr,nc))

        # 如果棋盘是空的，下在中间
        if not has_piece:
            mid = SIZE//2
            return [(mid,mid)]

        # 如果没有邻近空位（不太可能发生，除非棋盘满了），返回所有空位
        return list(moves) if moves else [(r,c) for r in range(SIZE) for c in range(SIZE) if board[r][c]==EMPTY]

    def _evaluate_board(self, board, who):
        # (保持不变) 整体评估：AI 分数 减去 玩家分数
        # 乘以一个略大于1的系数给对手分数，表示更倾向于阻止对手
        my_score    = self._calculate_player_score(board, AI)
        oppo_score  = self._calculate_player_score(board, PLAYER)

        # 如果一方已经获胜，给予极高/极低分
        if my_score   >= SCORE["FIVE"]: return SCORE["FIVE"] * 10
        if oppo_score >= SCORE["FIVE"]: return -SCORE["FIVE"] * 10

        return my_score - oppo_score * 1.1 # 稍微加大阻止对手的权重

    def _calculate_player_score(self, board, player):
        # (保持不变) 计算指定玩家的总分
        total_score = 0
        # 评估所有行、列、对角线
        # 优化：可以只评估包含该 player 棋子的线段
        evaluated_lines = set() # 避免重复计算

        # Rows
        for r in range(SIZE):
            line = board[r]
            total_score += self._evaluate_line(line, player) # 使用优化后的函数

        # Columns
        for c in range(SIZE):
            line = [board[r][c] for r in range(SIZE)]
            total_score += self._evaluate_line(line, player) # 使用优化后的函数

        # Diagonals (top-left to bottom-right)
        for i in range(-(SIZE - 1), SIZE):
            line = []
            for j in range(SIZE):
                r, c = j, j + i
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(board[r][c])
            if len(line) >= 5: # 至少需要5个才能形成五连
                 total_score += self._evaluate_line(line, player) # 使用优化后的函数

        # Diagonals (top-right to bottom-left)
        for i in range(SIZE * 2 - 1):
            line = []
            for j in range(SIZE):
                r, c = j, i - j
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(board[r][c])
            if len(line) >= 5:
                 total_score += self._evaluate_line(line, player) # 使用优化后的函数

        return total_score

    def _evaluate_line(self, line, player):
        """
        优化后的评估函数：直接遍历列表，不使用字符串转换。
        统计连续棋子数和两端情况。
        """
        score = 0
        n = len(line)
        opponent = PLAYER if player == AI else AI
        consecutive = 0 # 连续棋子数
        empty_ends = 0 # 两端空位数 (0, 1, or 2)
        in_pattern = False # 是否正在一个潜在的模式中

        for i in range(n):
            if line[i] == player:
                if not in_pattern:
                    in_pattern = True
                    consecutive = 1
                    # 检查左侧是空还是边界
                    empty_ends = 1 if (i == 0 or line[i-1] == EMPTY) else 0
                else:
                    consecutive += 1
            elif line[i] == EMPTY:
                if in_pattern:
                    # 模式结束，检查右侧是空还是边界
                    empty_ends += 1 if (i == n or line[i] == EMPTY) else 0 # 注意: 这里 line[i] == EMPTY
                    score += self._score_pattern(consecutive, empty_ends)
                    in_pattern = False
                    consecutive = 0
                    empty_ends = 0
            else: # opponent
                if in_pattern:
                    # 模式被对手阻断
                    # 检查右侧是否为空（被对手阻断不算空）
                    score += self._score_pattern(consecutive, empty_ends) # empty_ends 只能是 0 或 1
                    in_pattern = False
                    consecutive = 0
                    empty_ends = 0
                # 如果前面是空位，这个对手棋子可能挡住了之前的空位

        # 处理行尾的模式
        if in_pattern:
            # 检查右侧是空还是边界
            empty_ends += 1 # 到达末尾相当于右侧是"空"（未被对手阻断）
            score += self._score_pattern(consecutive, empty_ends)

        # --- 特殊模式处理 (跳子) ---
        # 活三的一种: P P E P / P E P P
        # 活二的一种: P E P
        # 这些在上面的简单连续计数中可能被低估，需要额外检查
        # P P E P
        for i in range(n - 3):
            if line[i:i+4] == [player, player, EMPTY, player]:
                left_empty = (i > 0 and line[i-1] == EMPTY)
                right_empty = (i + 4 < n and line[i+4] == EMPTY)
                if left_empty and right_empty:
                     # O E P P E P E O -> 认为是活三潜力
                     score += SCORE["LIVE_THREE"] // 2 # 给一个稍低的分，避免重复计算
                elif left_empty or right_empty:
                     score += SCORE["SLEEP_THREE"]

        # P E P P
        for i in range(n - 3):
             if line[i:i+4] == [player, EMPTY, player, player]:
                left_empty = (i > 0 and line[i-1] == EMPTY)
                right_empty = (i + 4 < n and line[i+4] == EMPTY)
                if left_empty and right_empty:
                    score += SCORE["LIVE_THREE"] // 2
                elif left_empty or right_empty:
                     score += SCORE["SLEEP_THREE"]

        # P E P (活二的一种) - 简单版本只看连续，这里补充
        # for i in range(n - 2):
        #     if line[i:i+3] == [player, EMPTY, player]:
        #         left_empty = (i > 0 and line[i-1] == EMPTY)
        #         right_empty = (i + 3 < n and line[i+3] == EMPTY)
        #         if left_empty and right_empty:
        #             # E P E P E -> 算活二潜力
        #             score += SCORE["LIVE_TWO"] // 2 # 避免与连续两个的活二重叠过多

        return score

    def _score_pattern(self, count, open_ends):
        """根据连续棋子数和两端空位数给出分数"""
        if count >= 5:
            return SCORE["FIVE"] # 五连

        if open_ends == 0: # 两端都被堵死或到达边界（且被对方棋子紧邻）
            return 0 # 死棋，没有发展潜力

        if count == 4:
            if open_ends == 2: return SCORE["LIVE_FOUR"] # 活四 E PPPP E
            if open_ends == 1: return SCORE["RUSH_FOUR"] # 冲四 O PPPP E or E PPPP O
        if count == 3:
            if open_ends == 2: return SCORE["LIVE_THREE"] # 活三 E PPP E
            if open_ends == 1: return SCORE["SLEEP_THREE"] # 眠三 O PPP E or E PPP O
        if count == 2:
            if open_ends == 2: return SCORE["LIVE_TWO"] # 活二 E PP E
            if open_ends == 1: return SCORE["SLEEP_TWO"] # 眠二 O PP E or E PP O
        # if count == 1:
        #     if open_ends == 2: return SCORE["LIVE_ONE"] # 活一 E P E (价值很低，可以忽略或给极小分)
        #     if open_ends == 1: return SCORE["SLEEP_ONE"] # 眠一 O P E or E P O (价值更低)

        return 0 # 其他情况（count < 1 或 open_ends < 0 不会发生）


# --- Flask 路由 (保持不变) ---
@app.route('/', methods=['GET'])
def index():
    # 确保从正确的基础目录提供文件
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json or {}
    if 'board' not in data or 'depth' not in data:
        return jsonify({"error":"缺少 board 或 depth 参数"}), 400

    board = data['board']
    depth = int(data['depth'])
    # 从请求获取 'threads' 参数，现在代表 'workers'
    workers = int(data.get('threads', os.cpu_count() or 1)) # 前端仍叫 threads

    # --- 输入验证 ---
    if not (1 <= depth <= 12): # 深度限制，太深会非常慢
        return jsonify({"error":"搜索深度 (depth) 必须在 1 到 12 之间"}), 400
    if not (1 <= workers <= 100): # 工作单元数限制
        return jsonify({"error":"并行工作单元数 (threads/workers) 必须在 1 到 100 之间"}), 400
    if not (isinstance(board, list) and len(board) == SIZE and
            all(isinstance(row, list) and len(row) == SIZE for row in board) and
            all(cell in [EMPTY, PLAYER, AI] for row in board for cell in row)):
        return jsonify({"error":"棋盘 (board) 格式错误或包含无效值"}), 400

    # --- 创建 AI 实例并获取走法 ---
    ai = GomokuAI(board, depth, workers)
    try:
        move = ai.find_best_move()
        if move is None: # AI未能找到合适的移动
             return jsonify({"error":"AI 未能确定下一步走法"}), 500
        return jsonify({"move": move})
    except Exception as e:
        # 记录详细错误日志
        import traceback
        print("="*20 + " AI 计算出错 " + "="*20)
        print(f"请求数据: board={board}, depth={depth}, workers={workers}")
        traceback.print_exc()
        print("="*50)
        # 返回通用错误给前端
        return jsonify({"error": f"AI 内部计算错误: {type(e).__name__}"}), 500


# --- 启动 (保持不变) ---
if __name__=='__main__':
    port = 5000
    print(f"=== Gomoku AI Server (Multiprocessing) 启动，端口 {port} ===")
    print(f"本机 CPU 核心数: {os.cpu_count() or '未知'}")
    # 注意：Flask 开发服务器是单线程的，对于真正的并发处理部署时应使用 Gunicorn/uWSGI 等
    # 但这里的并行是由 ProcessPoolExecutor 处理的，所以开发服务器也能看到多进程效果
    app.run(host='127.0.0.1', port=port, debug=False, threaded=False) # 明确禁用 Flask 的线程模式