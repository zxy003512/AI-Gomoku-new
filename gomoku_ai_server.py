# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import copy # 用于深拷贝棋盘

app = Flask(__name__)
# 允许来自任何源的请求（用于简单的本地开发）
# 对于生产环境，应将其限制为实际前端的源
CORS(app)

# --- 常量 ---
SIZE = 15
PLAYER = 1
AI = 2
EMPTY = 0

# 评分常量 (根据难度/风格调整)
# 确保这些值足够大以区分优先级
# 稍微降低 Rush Four 和 Live Three 的分数，增加 Sleep Three 的分数，以更注重防守
SCORE = {
    "FIVE": 100000000, # 绝对优先
    "LIVE_FOUR": 10000000,
    "RUSH_FOUR": 500000,  # 冲四也很重要，但活四优先度更高
    "LIVE_THREE": 50000, # 活三
    "SLEEP_THREE": 5000,  # 眠三价值提高
    "LIVE_TWO": 500,
    "SLEEP_TWO": 100,
    "LIVE_ONE": 10,
    "SLEEP_ONE": 1,
    "CENTER_BONUS": 1  # 中心的微小奖励 (可以忽略或微调)
}

# --- 游戏逻辑 (保持不变) ---

def is_valid(board, x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    # 检查 (y, x) 是否在棋盘内且是该玩家的棋子
    if not (0 <= y < SIZE and 0 <= x < SIZE and board[y][x] == player):
        return False

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平, 垂直, 右下斜, 右上斜
    for dx, dy in directions:
        count = 1
        # 正方向检查
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if not is_valid(board, nx, ny) or board[ny][nx] != player:
                break
            count += 1
        # 反方向检查
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if not is_valid(board, nx, ny) or board[ny][nx] != player:
                break
            count += 1
        if count >= 5:
            return True
    return False

# --- AI 核心逻辑 (Minimax, 评估) ---

class GomokuAI:
    # Zobrist 哈希表初始化移到类级别，确保所有实例共享
    zobrist_table = [[[random.randint(1, 2**64 - 1) for _ in range(3)]
                       for _ in range(SIZE)]
                      for _ in range(SIZE)]
    # 初始化一个空的棋盘哈希值 (可以根据需要计算，但0是常用初始值)
    empty_board_hash = 0
    # (可以预计算空棋盘哈希，但如果棋盘总是变化的，从0开始异或即可)
    # for r in range(SIZE):
    #     for c in range(SIZE):
    #         # Assuming EMPTY is 0, no need to XOR for empty cells if base hash is 0
    #         pass # No operation needed if initial hash is 0 and EMPTY is 0

    def __init__(self, board, depth):
        self.initial_board = [row[:] for row in board] # 保存初始棋盘状态
        self.depth = depth
        # !!! 关键修改：置换表在 AI 实例之间共享 !!!
        # 我们将在 find_best_move 中创建和传递它
        # self.transposition_table = {} # 不在这里初始化

    def _hash_board(self, board):
        """计算给定棋盘状态的哈希值"""
        h = 0 # 从空棋盘哈希开始
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != EMPTY:
                    player_idx = board[r][c] # 1 or 2
                    h ^= self.zobrist_table[r][c][player_idx]
        return h

    def _update_hash(self, current_hash, r, c, player):
        """更新哈希值（用于落子或撤销）"""
        # player_idx = 0 (EMPTY), 1 (PLAYER), 2 (AI)
        # Zobrist Hashing 的关键是 XOR 操作
        # 落子: hash ^ Zobrist[r][c][player]
        # 撤销: hash ^ Zobrist[r][c][player] (因为 a ^ b ^ b = a)
        return current_hash ^ self.zobrist_table[r][c][player]

    def find_best_move(self):
        start_time = time.time()
        best_score = -math.inf
        best_move = None
        current_board = [row[:] for row in self.initial_board] # 工作副本

        # 1. 检查AI能否一步获胜
        immediate_win_move = self._find_immediate_win(current_board, AI)
        if immediate_win_move:
            print(f"AI found immediate win at {immediate_win_move}")
            return {"y": immediate_win_move[0], "x": immediate_win_move[1]}

        # 2. 检查玩家能否一步获胜并阻止
        immediate_block_move = self._find_immediate_win(current_board, PLAYER)
        if immediate_block_move:
            print(f"AI blocking player win at {immediate_block_move}")
            # 确保阻挡位置是空的 (理论上应该是，因为是基于空位生成的)
            if current_board[immediate_block_move[0]][immediate_block_move[1]] == EMPTY:
                 return {"y": immediate_block_move[0], "x": immediate_block_move[1]}
            else:
                 print(f"Warning: Block move target {immediate_block_move} not empty. Proceeding with search.")


        # 3. Minimax 搜索
        print(f"Starting Minimax search with depth: {self.depth}")
        alpha = -math.inf
        beta = math.inf

        # 生成候选走法，并进行初步排序
        moves = self._generate_moves(current_board, AI) # 始终基于当前棋盘状态生成
        if not moves:
             print("Error: No valid moves found!")
             # 尝试找到棋盘上任何一个空位作为备选
             for r in range(SIZE):
                 for c in range(SIZE):
                     if current_board[r][c] == EMPTY:
                         print(f"Fallback: Returning first empty spot ({r},{c})")
                         return {"y": r, "x": c}
             return None # 如果棋盘满了

        scored_moves = []
        for r, c in moves:
            current_board[r][c] = AI
            # 使用快速评估进行排序
            score = self._evaluate_board(current_board, AI) # 评估AI落子后的局面
            current_board[r][c] = EMPTY # 撤销
            scored_moves.append(((r, c), score))

        # 按启发式评估分数降序排序，优先搜索好的走法
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        sorted_moves = [move for move, score in scored_moves]

        # --- 多线程评估顶层走法 ---
        # !!! 关键修改：创建共享置换表 !!!
        transposition_table = {}
        initial_hash = self._hash_board(current_board) # 计算初始棋盘哈希

        # 根据CPU核心数动态设置线程数，并设置上限(例如4或8)，避免在Vercel过多线程
        max_workers = max(1, min(os.cpu_count() or 1, 4)) # 限制最大线程数为4
        print(f"Using {max_workers} worker threads.")

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 对排序后的走法提交任务
            for r, c in sorted_moves:
                # !!! 关键修改：传递棋盘副本和共享置换表 !!!
                board_copy = [row[:] for row in current_board]
                board_copy[r][c] = AI # 在副本上落子
                move_hash = self._update_hash(initial_hash, r, c, AI) # 更新哈希

                # 提交任务到线程池，注意传递 board_copy, move_hash, 和共享的 transposition_table
                # 深度减1，因为AI已经走了一步，下一层是对手(Minimizing)
                future = executor.submit(self._minimax_thread_entry,
                                         board_copy, self.depth - 1, False, alpha, beta, move_hash, transposition_table)
                futures.append((future, (r, c)))

            # 按提交顺序（也是排序后的优先级顺序）处理结果
            processed_moves = 0
            for future, move in futures:
                try:
                    score = future.result() # 获取线程计算结果
                    processed_moves += 1
                    print(f"  Move ({move[0]},{move[1]}) evaluated score: {score}")

                    # 这是AI (Maximizer) 的回合，我们希望分数越高越好
                    if score > best_score:
                        best_score = score
                        best_move = move
                        print(f"  New best move: ({move[0]},{move[1]}) with score {score}")

                    # Alpha-Beta 剪枝 (在主线程中更新 alpha)
                    # 因为线程是并发的，这里的 alpha 更新可能不如纯串行有效，
                    # 但对于顶层来说，找到一个足够好的分数后可以提前考虑 (虽然不能停止其他线程)
                    alpha = max(alpha, score)
                    # 注意：在并发模型中，我们不能在这里基于 alpha > beta 来停止提交任务或取消 futures，
                    # 因为其他线程可能已经开始或快要完成了。我们只能在收集结果时应用这个逻辑。

                except Exception as exc:
                    print(f"Move ({move[0]},{move[1]}) generated an exception: {exc}")
                    # 可以选择给这个走法一个极低的分数，或者忽略它

        if not best_move and sorted_moves:
            print("Minimax didn't find a definitive best move (or all resulted in errors), picking first sorted move.")
            best_move = sorted_moves[0] # Fallback to the heuristically best move
        elif not best_move:
             print("Error: No valid moves could be determined after search!")
             # 再次尝试寻找空位
             for r in range(SIZE):
                 for c in range(SIZE):
                     if current_board[r][c] == EMPTY:
                         print(f"Fallback: Returning first empty spot ({r},{c})")
                         return {"y": r, "x": c}
             return None


        end_time = time.time()
        print(f"AI Calculation time: {end_time - start_time:.2f} seconds ({processed_moves}/{len(sorted_moves)} moves evaluated)")
        print(f"AI chose move: {best_move} with score: {best_score}")
        return {"y": best_move[0], "x": best_move[1]}

    def _find_immediate_win(self, board, player):
        """查找指定玩家可以一步获胜的位置"""
        potential_moves = self._generate_moves(board, player) # 生成该玩家的潜在落子点
        for r, c in potential_moves:
            if board[r][c] == EMPTY: # 确保是空位
                board[r][c] = player # 试下
                if check_win(board, player, c, r):
                    board[r][c] = EMPTY # 撤销
                    return (r, c) # 找到立即获胜/需阻止的位置
                board[r][c] = EMPTY # 撤销
        return None

    # 这个函数现在是线程的入口点
    def _minimax_thread_entry(self, board_state, depth, is_maximizing, alpha, beta, board_hash, transposition_table):
        """线程执行的 Minimax 函数，使用传入的棋盘状态和置换表"""
        # 注意：这个函数需要能够修改它自己的 board_state 副本
        # 它不应该直接修改 self.initial_board
        return self._minimax_memo(board_state, depth, is_maximizing, alpha, beta, board_hash, transposition_table)


    # Minimax 核心递归函数 (带 Alpha-Beta 和置换表)
    # !!! 关键修改：接收 board_state 和 transposition_table 作为参数 !!!
    def _minimax_memo(self, board_state, depth, is_maximizing, alpha, beta, board_hash, transposition_table):
        """
        Minimax 核心递归函数。
        :param board_state: 当前棋盘状态 (应该是副本，会被修改)
        :param depth: 当前搜索深度
        :param is_maximizing: 当前是最大化玩家(AI)还是最小化玩家(Player)
        :param alpha: Alpha 值
        :param beta: Beta 值
        :param board_hash: 当前棋盘状态的哈希值
        :param transposition_table: 共享的置换表
        :return: 当前状态的评估分数
        """
        state_key = (board_hash, depth, is_maximizing)
        if state_key in transposition_table:
            return transposition_table[state_key]

        # 检查是否达到叶节点或游戏结束状态
        # (可以在这里添加 check_win 的检查，如果某方已赢，直接返回极值分数)
        # 例如: if check_win(board_state, AI, ...): return SCORE["FIVE"] # 如果 AI 在此状态赢了
        #       if check_win(board_state, PLAYER, ...): return -SCORE["FIVE"] # 如果 Player 在此状态赢了
        # 注意：check_win 需要最后落子的坐标，这里可能没有，需要更复杂的胜利检查或依赖评估函数给出足够大的分数

        if depth == 0:
            # 评估当前局面分数 (相对于 AI)
            # 注意：评估函数应始终从 AI 的角度出发
            score = self._evaluate_board(board_state, AI)
            transposition_table[state_key] = score
            return score

        # 生成当前玩家的走法
        current_player = AI if is_maximizing else PLAYER
        moves = self._generate_moves(board_state, current_player)

        # 如果没有可行的走法（罕见，除非棋盘满了或被完全围住）
        if not moves:
            # 在这种情况下通常是平局或者某一方已被迫无路可走
            score = self.evaluate_final_state(board_state) # 评估最终状态
            transposition_table[state_key] = score
            return score


        if is_maximizing:  # AI (Max player)
            max_eval = -math.inf
            # 可以考虑对这里的 moves 再进行一次排序 (代价较高)
            for r, c in moves:
                # 落子并更新哈希
                board_state[r][c] = AI
                new_hash = self._update_hash(board_hash, r, c, AI)

                # 递归调用
                eval_score = self._minimax_memo(board_state, depth - 1, False, alpha, beta, new_hash, transposition_table)

                # 撤销落子并恢复哈希
                board_state[r][c] = EMPTY
                # board_hash = self._update_hash(new_hash, r, c, AI) # 恢复哈希 (其实不需要，因为 board_hash 是上层的)

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta 剪枝
            transposition_table[state_key] = max_eval
            return max_eval
        else:  # Player (Min player)
            min_eval = math.inf
            # 可以考虑对这里的 moves 再进行一次排序 (代价较高)
            for r, c in moves:
                # 落子并更新哈希
                board_state[r][c] = PLAYER
                new_hash = self._update_hash(board_hash, r, c, PLAYER)

                # 递归调用
                eval_score = self._minimax_memo(board_state, depth - 1, True, alpha, beta, new_hash, transposition_table)

                # 撤销落子并恢复哈希
                board_state[r][c] = EMPTY
                # board_hash = self._update_hash(new_hash, r, c, PLAYER) # 恢复哈希

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha 剪枝
            transposition_table[state_key] = min_eval
            return min_eval

    def evaluate_final_state(self, board_state):
        """评估没有更多合法走法时的状态（可能是平局或一方被困）"""
        # 在此简化版本中，我们只返回一个中性分数（0），
        # 但更复杂的评估可以检查是否有一方实际上赢了，即使没有下最后一步。
        # 暂时使用基础评估函数，如果棋盘满了它应该能反映局面。
        return self._evaluate_board(board_state, AI)


    # 启发式候选走法生成
    def _generate_moves(self, board, player_to_move):
        """
        生成候选走法。只考虑已有棋子周围一定范围内的空位。
        优化：如果棋盘为空，则只返回中心点。
        添加更严格的筛选条件，例如必须紧邻棋子。
        """
        moves = set()
        has_pieces = False
        neighbor_radius = 1 # 只考虑紧邻的位置 (radius=1)
        # 也可以考虑 radius=2，但在性能敏感时 radius=1 更好
        # radius = 2

        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != EMPTY:
                    has_pieces = True
                    # 检查当前棋子周围 neighbor_radius 范围内的空位
                    for dr in range(-neighbor_radius, neighbor_radius + 1):
                        for dc in range(-neighbor_radius, neighbor_radius + 1):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if is_valid(board, nc, nr) and board[nr][nc] == EMPTY:
                                moves.add((nr, nc))

        # 如果棋盘上还没有棋子，AI第一步下在天元
        if not has_pieces:
            center = SIZE // 2
            if is_valid(board, center, center) and board[center][center] == EMPTY:
                return [(center, center)]
            else: # 如果中心被占了（不太可能在空棋盘发生），随便找个空位
                 for r in range(SIZE):
                    for c in range(SIZE):
                        if board[r][c] == EMPTY: return [(r,c)]
                 return [] # 满了


        # 进一步优化：只返回有"价值"的邻近点
        # 例如，可以添加一个检查，如果一个空位周围没有任何棋子，则不考虑它
        # (当前的实现已经隐式地做到了这一点，因为我们只从有棋子的地方扩展)

        # 再增加一层筛选：考虑“防御性”和“进攻性”候选点
        # - AI自己的活三、活四、冲四点
        # - 对手的活三、活四、冲四点 (必须防守的点)
        # 这个筛选比较复杂，暂时保持仅邻近策略，依赖评估函数和搜索

        if not moves and has_pieces:
            # 如果有棋子但周围没有空位 (棋盘快满了或特殊情况)
            # 此时应该搜索整个棋盘找剩余的空位
            print("Warning: No neighboring empty cells found, searching all empty cells.")
            all_empty = []
            for r in range(SIZE):
                for c in range(SIZE):
                    if board[r][c] == EMPTY:
                        all_empty.append((r,c))
            return all_empty

        return list(moves)


    # --- 评估函数 ---
    def _evaluate_board(self, board, player_perspective):
        """
        评估整个棋盘的分数。始终从 AI (PLAYER=2) 的角度进行评估。
        分数 = AI 的分数 - 对手的分数 * 阻击系数
        """
        ai_player = AI
        opponent_player = PLAYER

        ai_score = self._calculate_score_for_player(board, ai_player)
        opponent_score = self._calculate_score_for_player(board, opponent_player)

        # 如果AI或对手已经获胜，直接返回极值
        if ai_score >= SCORE["FIVE"]: return SCORE["FIVE"] * 10 # 确保胜利分数最高
        if opponent_score >= SCORE["FIVE"]: return -SCORE["FIVE"] * 10 # 对手胜利分数最低

        # 调整对手分数的权重，表示防守的重要性
        # 这个系数可以调整，大于1表示更注重防守
        block_factor = 1.1
        final_score = ai_score - opponent_score * block_factor

        # 添加中心位置奖励 (可选项)
        # center = SIZE // 2
        # if board[center][center] == ai_player:
        #     final_score += SCORE["CENTER_BONUS"]
        # elif board[center][center] == opponent_player:
        #     final_score -= SCORE["CENTER_BONUS"]

        return final_score

    def _calculate_score_for_player(self, board, player):
        """计算指定玩家在当前棋盘上的总得分"""
        total_score = 0
        evaluated_lines = set() # 用于避免重复计算同一条线的不同部分

        # 评估所有行、列、斜线
        # 优化: 我们可以只评估包含最近一步棋子的线，但这在递归评估中不适用
        # 这里还是评估全盘

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
             # Vercel 部署可能对边界情况敏感，这里使用更明确的匹配
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

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json
    if not data or 'board' not in data or 'depth' not in data:
        return jsonify({"error": "Missing board or depth in request"}), 400

    try:
        board_state = data['board']
        search_depth = int(data['depth'])

        # --- 输入验证 ---
        if not isinstance(board_state, list) or len(board_state) != SIZE:
             return jsonify({"error": "Invalid board format: board must be a list"}), 400
        if not all(isinstance(row, list) and len(row) == SIZE for row in board_state):
            return jsonify({"error": f"Invalid board format: each row must be a list of size {SIZE}"}), 400
        # 检查棋盘内容是否合法 (0, 1, 2)
        valid_cells = {EMPTY, PLAYER, AI}
        for r in range(SIZE):
            for c in range(SIZE):
                if board_state[r][c] not in valid_cells:
                    return jsonify({"error": f"Invalid cell value at ({r},{c}): {board_state[r][c]}"}), 400

        if not isinstance(search_depth, int) or search_depth <= 0 or search_depth > 6: # 限制最大深度
            # 对于Vercel免费版，深度4可能已经是极限，深度5/6会非常慢或超时
            return jsonify({"error": "Invalid depth (must be between 1 and 6)"}), 400

        print(f"\nReceived request: Depth={search_depth}")
        # 记录请求开始时间
        request_start_time = time.time()

        # --- AI 计算 ---
        ai = GomokuAI(board_state, search_depth)
        best_move = ai.find_best_move()

        # 记录请求结束时间
        request_end_time = time.time()
        print(f"Total request processing time: {request_end_time - request_start_time:.2f} seconds")

        # --- 返回结果 ---
        if best_move and 'x' in best_move and 'y' in best_move:
             # 再次验证 AI 返回的移动是否在棋盘内且为空
             y, x = best_move['y'], best_move['x']
             if not (0 <= y < SIZE and 0 <= x < SIZE):
                  print(f"Error: AI returned out-of-bounds move: ({x},{y})")
                  return jsonify({"error": "AI returned an out-of-bounds move"}), 500
             if board_state[y][x] != EMPTY:
                 print(f"Error: AI returned move to non-empty cell: ({x},{y}) which contains {board_state[y][x]}")
                 # 尝试找一个备用空位
                 for r_idx in range(SIZE):
                     for c_idx in range(SIZE):
                         if board_state[r_idx][c_idx] == EMPTY:
                             print(f"Fallback: Returning first empty spot ({r_idx},{c_idx}) due to AI error")
                             return jsonify({"move": {"y": r_idx, "x": c_idx}})
                 return jsonify({"error": "AI returned move to non-empty cell, and no fallback found"}), 500

             return jsonify({"move": best_move})
        elif best_move is None and not any(EMPTY in row for row in board_state):
             print("Board is full, no move possible.")
             return jsonify({"error": "Board is full, no move possible"}), 400 # 返回客户端可以处理的错误
        else:
            # 如果AI未能找到走法但棋盘未满，这是内部错误
            print("Error: AI failed to determine a move, but board is not full.")
            # 尝试找一个备用空位
            for r in range(SIZE):
                 for c in range(SIZE):
                     if board_state[r][c] == EMPTY:
                         print(f"Fallback: Returning first empty spot ({r},{c}) due to AI internal error")
                         return jsonify({"move": {"y": r, "x": c}})
            return jsonify({"error": "AI failed to find a move (internal error, no empty spots found)"}), 500

    except Exception as e:
        # 捕获所有潜在错误，记录日志并返回通用错误
        import traceback
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc()) # 打印详细的错误堆栈
        return jsonify({"error": "An internal server error occurred during AI calculation."}), 500


if __name__ == '__main__':
    # 在本地运行时，可以开启 debug=True，但在生产或 Vercel 环境中应关闭
    # --- 修改：开启 debug 模式 ---
    app.run(host='0.0.0.0', port=5000, debug=True)
