# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import random
import traceback
import copy
# from functools import lru_cache # 在 Numba 环境外可能有用，但核心在 Numba

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from concurrent.futures import ProcessPoolExecutor, TimeoutError, wait, FIRST_COMPLETED

import numpy as np
import numba

# --- 常量 ---
SIZE = 20  # <<--- 重要：修改棋盘大小为 20x20
BOARD_SIZE = SIZE # 确保 SIZE 在 Numba 函数中可用

PLAYER = np.int8(1) # 人类玩家 (黑棋)
AI = np.int8(2)     # AI 玩家 (红棋/白棋，取决于前端渲染)
EMPTY = np.int8(0)

# --- 基础目录 ---
BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# --- Flask 应用 ---
app = Flask(__name__, static_folder=None) # 不使用 Flask 的静态文件夹，由路由处理
CORS(app) # 允许跨域请求

# --- 更精细和大幅增强的评分常量 ---
SCORE_FIVE = 1000000000       # 成五 (极大提高)
SCORE_LIVE_FOUR = 50000000    # 活四 (极大提高)
SCORE_RUSH_FOUR = 4000000     # 冲四 / 死四活三 (极大提高)
SCORE_LIVE_THREE = 3000000    # 活三 (极大提高)
SCORE_SLEEP_THREE = 200000    # 眠三 (提高)
SCORE_LIVE_TWO = 5000         # 活二 (提高)
SCORE_SLEEP_TWO = 300         # 眠二 (提高)
SCORE_LIVE_ONE = 10           # 活一
SCORE_SLEEP_ONE = 2           # 眠一

# 组合和威胁加分 (隐式包含在极高的单模式分数中，也可在评估函数中动态计算)
# 以下分数用于 _evaluate_board 中的额外判断逻辑
SCORE_DOUBLE_LIVE_THREE = SCORE_LIVE_FOUR * 0.9 # 双活三接近活四威胁
SCORE_LIVE_THREE_RUSH_FOUR = SCORE_LIVE_FOUR * 0.95 # 活三+冲四组合
SCORE_DOUBLE_RUSH_FOUR = SCORE_LIVE_FOUR * 1.0 # 双冲四理论必胜

# --- Zobrist Hashing (重新生成以匹配 20x20) ---
# 确保随机状态一致性以便调试 (可选)
# np.random.seed(0)
zobrist_table = np.random.randint(np.iinfo(np.uint64).min, np.iinfo(np.uint64).max,
                                  size=(BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint64)

# --- Transposition Table (置换表) ---
# 每个工作进程将拥有自己的 TT 副本。TT Entry 结构:
# { 'score': score, 'depth': depth, 'flag': flag, 'best_move': move }
# flag: 0=Exact (精确值), 1=Lower Bound (Alpha, 下界), 2=Upper Bound (Beta, 上界)
# 全局 TT 在此模型中不直接共享，由子进程各自维护

# --- Numba 加速核心计算 ---

@numba.njit("b1(int8[:,:], i8, i8, i8, i4)", cache=True, fastmath=True)
def _check_win_numba(board_arr: np.ndarray, player: np.int8, x: int, y: int, board_size: int) -> bool:
    """Numba 加速的胜利检查 (检查最后落子位置)"""
    # 边界检查
    if not (0 <= y < board_size and 0 <= x < board_size and board_arr[y, x] == player):
        return False
    # 四个方向: 水平, 垂直, 主对角线, 反对角线
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for dx, dy in directions:
        count = 1 # 包含当前子
        # 检查两个相反方向
        for sign in numba.int64([-1, 1]):
            for i in range(1, 5): # 最多再检查4个子
                nx, ny = x + i * dx * sign, y + i * dy * sign
                # 检查是否在棋盘内且是同色棋子
                if not (0 <= nx < board_size and 0 <= ny < board_size and board_arr[ny, nx] == player):
                    break # 遇到边界或非同色子则停止该方向
                count += 1
        # 如果任何方向达到5子或以上
        if count >= 5:
            return True
    return False

# 重写并大幅增强的行评估函数
@numba.njit(cache=True, fastmath=True)
def _evaluate_line_numba(line: np.ndarray, player: np.int8, opponent: np.int8) -> np.int64:
    """
    Numba 加速的单行评估函数 (核心增强)
    识别更复杂的模式并返回该行的总分数 (从 player 角度)
    """
    score = np.int64(0)
    n = len(line)
    if n < 5: return np.int64(0)

    empty = EMPTY
    live_four_count = 0
    rush_four_count = 0
    live_three_count = 0
    sleep_three_count = 0
    live_two_count = 0
    sleep_two_count = 0
    live_one_count = 0
    sleep_one_count = 0

    # --- 使用滑动窗口识别模式 (窗口大小 5 和 6) ---
    # 窗口大小为 5 用于识别中间有空位的模式
    for i in range(n - 4):
        window5 = line[i:i+5]
        p_count = np.sum(window5 == player)
        o_count = np.sum(window5 == opponent)
        e_count = 5 - p_count - o_count

        if o_count == 0: # 窗口内不能有对手棋子
            # --- 冲四 (XXXX. , XXX.X, XX.XX, X.XXX, .XXXX) ---
            if p_count == 4 and e_count == 1:
                # 检查窗口外两侧是否有空位形成活四的潜力
                left_empty = (i > 0 and line[i-1] == empty)
                right_empty = (i + 5 < n and line[i+5] == empty)
                # 严格冲四定义：至少一侧被挡或到边界，另一侧为空
                is_rush = False
                if window5[0] == empty: # .OOOO
                    if i==0 or line[i-1] == opponent: # 左侧到边界或被挡
                       if right_empty: is_rush = True
                    elif left_empty: # 左侧有空
                       if i+5==n or line[i+5] == opponent: is_rush = True # 右侧到边界或被挡
                       # else: # 两侧都有空，下面活四会处理
                elif window5[4] == empty: # OOOO.
                    if i+5==n or line[i+5] == opponent: # 右侧到边界或被挡
                        if left_empty: is_rush = True
                    elif right_empty: # 右侧有空
                        if i==0 or line[i-1] == opponent: is_rush = True # 左侧到边界或被挡
                        # else: # 两侧都有空，下面活四会处理
                else: # O.OOO, OO.OO, OOO.O
                    # 中间断开的冲四，只要两端有一端能延伸即可
                     if left_empty or right_empty: is_rush = True

                if is_rush:
                     rush_four_count += 1
            # --- 眠三 (OOO.. , .OOO. , ..OOO, O.O.O, OO..O, O..OO) ---
            elif p_count == 3 and e_count == 2:
                is_sleep3 = False
                # OOO.. (右侧必须被挡或边界)
                if window5[0]==player and window5[1]==player and window5[2]==player:
                    if (i > 0 and line[i-1] == empty) and \
                       (i+5==n or line[i+5] == opponent): is_sleep3 = True
                # ..OOO (左侧必须被挡或边界)
                elif window5[2]==player and window5[3]==player and window5[4]==player:
                     if (i+5 < n and line[i+5] == empty) and \
                        (i==0 or line[i-1] == opponent): is_sleep3 = True
                # .OOO. (两端至少一个被挡)
                elif window5[1]==player and window5[2]==player and window5[3]==player:
                     left_blocked = (i==0 or line[i-1] == opponent)
                     right_blocked = (i+5==n or line[i+5] == opponent)
                     if left_blocked or right_blocked: is_sleep3 = True # 只要有一边被挡就是眠三（活三下面处理）
                # O.O.O (两端必须被挡)
                elif window5[0]==player and window5[2]==player and window5[4]==player:
                     if (i==0 or line[i-1] == opponent) and \
                        (i+5==n or line[i+5] == opponent): is_sleep3 = True
                # OO..O (左侧需空，右侧被挡)
                elif window5[0]==player and window5[1]==player and window5[4]==player:
                    if (i > 0 and line[i-1] == empty) and \
                       (i+5==n or line[i+5] == opponent): is_sleep3 = True
                # O..OO (右侧需空，左侧被挡)
                elif window5[0]==player and window5[3]==player and window5[4]==player:
                    if (i+5 < n and line[i+5] == empty) and \
                       (i==0 or line[i-1] == opponent): is_sleep3 = True

                if is_sleep3:
                    sleep_three_count += 1

        # 对方的棋子，也需要评估来计算防守价值（虽然最后分数是 player - opponent）
        # 这里简单跳过，最终评估函数会计算双方分数
        # else: # o_count > 0
        #    pass

    # 窗口大小为 6 用于识别活棋 (两端必须为空)
    for i in range(n - 5):
        window6 = line[i:i+6]
        p_count = np.sum(window6 == player)
        o_count = np.sum(window6 == opponent)

        # --- 成五 (忽略，由 check_win 处理或在评估函数中赋予极大值) ---
        # if p_count == 5: # XXXXX? or ?XXXXX
        #     if window6[0] == player or window6[5] == player:
        #          score += SCORE_FIVE # 应该在更高层处理

        # 必须两端为空才能是 "活" 棋
        if window6[0] == empty and window6[5] == empty and o_count == 0:
            # --- 活四 (.OOOO.) ---
            if p_count == 4:
                live_four_count += 1
            # --- 活三 (.OOO..) or (..OOO.) ---
            elif p_count == 3:
                # 检查是否是连续的三个 .OOO.. or ..OOO.
                if (window6[1]==player and window6[2]==player and window6[3]==player) or \
                   (window6[2]==player and window6[3]==player and window6[4]==player):
                    live_three_count += 1
                 # 检查跳活三 .O.OO. or .OO.O.
                elif (window6[1]==player and window6[3]==player and window6[4]==player) or \
                     (window6[1]==player and window6[2]==player and window6[4]==player):
                     live_three_count += 1 # 也算活三
            # --- 活二 (.OO...) or (..OO..) or (...OO.) ---
            elif p_count == 2:
                 # .OO...
                 if window6[1]==player and window6[2]==player: live_two_count += 1
                 # ..OO..
                 elif window6[2]==player and window6[3]==player: live_two_count += 1
                 # ...OO.
                 elif window6[3]==player and window6[4]==player: live_two_count += 1
                 # .O.O..
                 elif window6[1]==player and window6[3]==player: live_two_count += 1
                 # ..O.O.
                 elif window6[2]==player and window6[4]==player: live_two_count += 1
                 # .O..O. (这个比较弱，算不算活二？算上)
                 elif window6[1]==player and window6[4]==player: live_two_count += 1


    # --- 眠二和眠一的简单计数 (使用更简单的方法) ---
    # 统计所有长度为 2 和 1 的棋段，根据两端情况判断是活是眠
    i = 0
    while i < n:
        if line[i] == player:
            count = 0
            start = i
            while i < n and line[i] == player:
                count += 1
                i += 1
            end = i - 1

            left_open = (start > 0 and line[start - 1] == empty)
            right_open = (end < n - 1 and line[end + 1] == empty)
            open_ends = (1 if left_open else 0) + (1 if right_open else 0)

            if count == 2 and open_ends == 1: # 确保不与活二重复计数
                # 检查是否已经被 window6 计为活二 (.OO... etc)
                is_counted_live = False
                if left_open and start > 0 and start + 3 < n and line[start+2]==empty and line[start+3]==empty: # .OO..
                    pass # Likely counted as live two
                elif right_open and end < n - 1 and end - 3 >= 0 and line[end-2]==empty and line[end-3]==empty: # ..OO.
                    pass # Likely counted as live two
                else:
                     sleep_two_count += 1

            elif count == 1 and open_ends == 2:
                live_one_count += 1
            elif count == 1 and open_ends == 1:
                sleep_one_count += 1
        else:
            i += 1


    # --- 最终组合评分 ---
    # 注意：单个模式的分数已经很高，这里主要是处理特殊组合的额外加分或判定
    if live_four_count > 0 or rush_four_count >= 2: # 有一个活四或者两个冲四基本获胜
         score += SCORE_FIVE * 0.95 # 接近成五的分数

    elif live_three_count >= 2: # 双活三
         score += SCORE_DOUBLE_LIVE_THREE

    elif live_three_count == 1 and rush_four_count == 1: # 活三带冲四
         score += SCORE_LIVE_THREE_RUSH_FOUR

    # 基础分累加
    score += live_four_count * SCORE_LIVE_FOUR
    score += rush_four_count * SCORE_RUSH_FOUR
    score += live_three_count * SCORE_LIVE_THREE
    # 对眠三进行裁剪，避免过高估计其价值
    score += sleep_three_count * SCORE_SLEEP_THREE if live_four_count==0 and rush_four_count==0 else sleep_three_count * SCORE_SLEEP_THREE * 0.5
    # 对活二进行裁剪
    score += live_two_count * SCORE_LIVE_TWO if live_four_count==0 and rush_four_count==0 and live_three_count==0 else live_two_count * SCORE_LIVE_TWO * 0.5
    score += sleep_two_count * SCORE_SLEEP_TWO
    score += live_one_count * SCORE_LIVE_ONE
    score += sleep_one_count * SCORE_SLEEP_ONE

    # 确保分数不会溢出 int64 (虽然概率很小)
    if score > np.iinfo(np.int64).max: score = np.iinfo(np.int64).max
    if score < np.iinfo(np.int64).min: score = np.iinfo(np.int64).min

    return score


@numba.njit(cache=True, fastmath=True)
def _calculate_player_score_numba(board_arr: np.ndarray, player: np.int8, opponent: np.int8, board_size: int) -> np.int64:
    """Numba 加速的玩家总分计算 (调用 _evaluate_line_numba)"""
    total_score = np.int64(0)
    # 检查是否已经有一方成五 (优化：如果已有五，直接返回极大/极小值)
    # 这个检查应该在更高层完成，这里只计算分数

    # Rows & Columns
    for i in range(board_size):
        total_score += _evaluate_line_numba(board_arr[i, :], player, opponent) # 评估行
        total_score += _evaluate_line_numba(board_arr[:, i], player, opponent) # 评估列

    # Diagonals (优化对角线提取)
    for offset in range(-(board_size - 5), board_size - 4): # 只需要长度至少为5的对角线
        # 主对角线 (top-left to bottom-right)
        diag = np.diag(board_arr, k=offset)
        if len(diag) >= 5:
             total_score += _evaluate_line_numba(diag, player, opponent)

        # 反对角线 (top-right to bottom-left)
        # 通过翻转棋盘再取主对角线实现
        anti_diag = np.diag(np.fliplr(board_arr), k=offset)
        if len(anti_diag) >= 5:
             total_score += _evaluate_line_numba(anti_diag, player, opponent)

    return total_score

# --- 公共工具函数 ---
# 公共胜利检查接口 (可以缓存结果，但 Numba njit 本身有缓存)
# @lru_cache(maxsize=None) # 在 Python 层面缓存效果有限，因为 board_arr 是可变对象
def check_win(board_arr, player, x, y):
    """公共胜利检查接口"""
    if not isinstance(board_arr, np.ndarray):
        board_arr = np.array(board_arr, dtype=np.int8)
    player_np = np.int8(player)
    # 直接调用 Numba 函数
    return _check_win_numba(board_arr, player_np, x, y, BOARD_SIZE)


# --- AI 核心类 (大幅修改) ---
class GomokuAI:
    def __init__(self, board, depth, max_workers=None):
        self.initial_board_arr = np.array(board, dtype=np.int8)
        self.depth = max(1, depth) # 确保深度至少为 1
        # 限制最大进程数，避免过多开销, 尤其对于高计算量的评估函数
        self.max_workers = max(1, min(max_workers if max_workers is not None else (os.cpu_count() or 1), os.cpu_count() or 1, 8))
        self.quiescence_depth = max(1, depth // 2) # 静态威胁搜索的额外深度，设为主深度的一半
        self.start_time = 0
        self.time_limit_per_move = 600 # seconds (设置一个宽松的默认值，因为用户说不要时间限制)
        self.nodes_searched = 0
        self.tt_hits = 0
        self.q_nodes_searched = 0 # 静态搜索节点计数
        # 每个 AI 实例可以有自己的 TT，但多进程模式下子进程通常创建新的
        # self.transposition_table = {}

    def _hash_board(self, board_arr):
        """计算当前棋盘的 Zobrist 哈希值"""
        h = np.uint64(0)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = board_arr[r, c]
                if piece != EMPTY:
                    # piece 是 0, 1(PLAYER), 2(AI)
                    # zobrist_table 的第三维索引是 piece (0, 1, 2)
                    h ^= zobrist_table[r, c, piece]
        return h

    def _update_hash(self, h, r, c, player, old_player=EMPTY):
        """增量更新 Zobrist 哈希值"""
        if old_player != EMPTY:
            h ^= zobrist_table[r, c, old_player] # 移除旧棋子
        if player != EMPTY:
            h ^= zobrist_table[r, c, player] # 加入新棋子
        return h

    def _evaluate_board(self, board_arr: np.ndarray, last_move=None) -> float:
        """
        评估整个棋盘状态 (从 AI 角度)
        包含更复杂的威胁评估和防守加权
        last_move: (r, c, player) 可选，用于快速检查胜负
        """
        # 快速胜负检查 (如果提供了最后一步)
        if last_move:
            lr, lc, lplayer = last_move
            if _check_win_numba(board_arr, lplayer, lc, lr, BOARD_SIZE):
                return SCORE_FIVE * 10 if lplayer == AI else -SCORE_FIVE * 10

        # 完整计算双方分数
        ai_score = _calculate_player_score_numba(board_arr, AI, PLAYER, BOARD_SIZE)
        player_score = _calculate_player_score_numba(board_arr, PLAYER, AI, BOARD_SIZE)

        # --- 胜负判断 (再次确认，防止_calculate中的近似) ---
        if ai_score >= SCORE_FIVE: return float(SCORE_FIVE * 10) # AI 胜
        if player_score >= SCORE_FIVE: return float(-SCORE_FIVE * 10) # Player 胜

        # --- 组合威胁评估 (部分已包含在单项高分中) ---
        # 显式检查一些高威胁组合
        if player_score >= SCORE_LIVE_FOUR or \
           (player_score >= SCORE_DOUBLE_LIVE_THREE and ai_score < SCORE_LIVE_FOUR) or \
           (player_score >= SCORE_LIVE_THREE_RUSH_FOUR and ai_score < SCORE_LIVE_FOUR):
             # 如果对手有极高威胁（活四或多种组合杀），且AI自身没有同等级威胁，给予极大负分强制防守
             return float(-SCORE_LIVE_FOUR * 5) # 必须防守这类威胁

        if ai_score >= SCORE_LIVE_FOUR or \
           ai_score >= SCORE_DOUBLE_LIVE_THREE or \
           ai_score >= SCORE_LIVE_THREE_RUSH_FOUR:
            # 如果 AI 有极高威胁，给予极高正分
            return float(SCORE_LIVE_FOUR * 5)

        # --- 基础评估: AI 分数 - 对手分数 * 防守权重 ---
        # 防守权重可以略大于 1，表示防守更重要，特别是在均势时
        defense_weight = 1.1 # 稍微重视防守
        final_score = float(ai_score - player_score * defense_weight)

        # --- 加入棋子位置价值 (可选，给靠近中心的棋子微小加分) ---
        # center_bonus = 0
        # center = BOARD_SIZE // 2
        # for r in range(BOARD_SIZE):
        #     for c in range(BOARD_SIZE):
        #         if board_arr[r, c] == AI:
        #             center_bonus += max(0, center - max(abs(r - center), abs(c - center)))
        #         elif board_arr[r, c] == PLAYER:
        #             center_bonus -= max(0, center - max(abs(r - center), abs(c - center))) * defense_weight
        # final_score += center_bonus * 0.1 # 给予微小的中心权重

        return final_score


    def _generate_moves(self, board_arr: np.ndarray, player: np.int8, for_quiescence=False):
        """
        生成候选着法 (优化)
        - for_quiescence: True 表示只生成战术性着法 (用于静态搜索)
        """
        moves = []
        immediate_wins = []
        opponent_wins_next = [] # 对手下一步能赢的点 (必须防守)
        ai_threats = [] # AI 的进攻点 (活四/冲四/活三)
        player_blocks = [] # 防守点 (挡住对方关键棋型)
        neighbor_moves = set() # 附近有棋子的空点
        has_piece = False # 棋盘上是否有棋子
        radius = 1 # 检查邻居范围 (可以根据棋局阶段调整，初期大，后期小?)

        opponent = PLAYER if player == AI else AI

        # --- 优先级 1: 查找己方能一步获胜的点 ---
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_arr[r, c] == EMPTY:
                    board_arr[r, c] = player
                    if _check_win_numba(board_arr, player, c, r, BOARD_SIZE):
                        immediate_wins.append((r, c))
                    board_arr[r, c] = EMPTY # 回溯
        if immediate_wins:
            # print(f"[Debug] Player {player} Immediate wins found: {immediate_wins}")
            return immediate_wins # 如果能赢，只考虑这些点

        # --- 优先级 2: 查找对方下一步能赢的点 (必须防守) ---
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_arr[r, c] == EMPTY:
                    board_arr[r, c] = opponent
                    if _check_win_numba(board_arr, opponent, c, r, BOARD_SIZE):
                        opponent_wins_next.append((r, c))
                    board_arr[r, c] = EMPTY # 回溯
        if opponent_wins_next:
            # print(f"[Debug] Player {player} Must block opponent wins: {opponent_wins_next}")
            return opponent_wins_next # 如果必须防守，只考虑这些点

        # --- 优先级 3: 生成威胁点和防守点 (启发式) ---
        potential_moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_arr[r, c] == EMPTY:
                    # 检查该点附近是否有棋子 (减少搜索范围)
                    has_neighbor_piece = False
                    for dr in range(-radius, radius + 1):
                       for dc in range(-radius, radius + 1):
                           if dr == 0 and dc == 0: continue
                           nr, nc = r + dr, c + dc
                           if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board_arr[nr, nc] != EMPTY:
                                has_neighbor_piece = True
                                break
                       if has_neighbor_piece: break

                    if has_neighbor_piece:
                        potential_moves.append((r, c))
                        has_piece = True # 标记棋盘上有棋子

        # 如果棋盘为空，下中间或附近
        if not has_piece:
            center_r, center_c = BOARD_SIZE // 2, BOARD_SIZE // 2
            # 随机下在中心 3x3 区域内
            offset_r = random.randint(-1, 1)
            offset_c = random.randint(-1, 1)
            cr, cc = center_r + offset_r, center_c + offset_c
            if 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board_arr[cr, cc] == EMPTY:
                 return [(cr, cc)]
            else: # 如果中心被占，返回第一个空位（备用）
                 return [(center_r, center_c)] if board_arr[center_r, center_c] == EMPTY else self._fallback_move(board_arr, single_move=True)


        # --- 评估潜在着法的价值 (生成进攻和防守点) ---
        scored_potential_moves = []
        for r, c in potential_moves:
            score_player = 0
            score_opponent = 0
            is_critical_defense = False

            # 评估我方落子价值
            board_arr[r, c] = player
            # 快速评估单点落子后的棋型变化 (更高效的方式是只评估包含该点的8条线)
            # 这里简化，使用全局评估的差值，但效率较低
            # score_player = self._evaluate_board(board_arr) # 效率太低
            # 改为只评估新产生的棋型分数
            score_player = self._quick_evaluate_move(board_arr, r, c, player)
            board_arr[r, c] = EMPTY

            # 评估对方在此落子价值 (即我方防守价值)
            board_arr[r, c] = opponent
            score_opponent = self._quick_evaluate_move(board_arr, r, c, opponent)
            board_arr[r, c] = EMPTY

            # 合并分数，给予进攻和防守不同的权重
            # 防守分数越高，说明这个点对对方越重要，我方占据的价值（防守价值）也越高
            # 进攻分数越高，说明这个点对我方越有利
            combined_score = score_player + score_opponent * 1.5 # 稍微侧重防守价值

            # 判断是否为关键防守点 (例如对方下此可形成活三以上)
            if score_opponent >= SCORE_LIVE_THREE:
                 is_critical_defense = True
                 combined_score += SCORE_LIVE_THREE # 大幅提高关键防守点优先级

            scored_potential_moves.append(((r, c), combined_score, score_player >= SCORE_LIVE_THREE, is_critical_defense))

        # --- 排序和筛选 ---
        # 优先级: 关键进攻点 > 关键防守点 > 其他邻近点
        scored_potential_moves.sort(key=lambda x: (x[2], x[3], x[1]), reverse=True)

        # 如果是静态搜索，只选取价值最高的几个威胁/防守点
        if for_quiescence:
            # 选择分数 > 眠三的点，或者选择前 N 个点
            q_moves = [m for m, score, is_threat, is_defense in scored_potential_moves if score >= SCORE_SLEEP_THREE * 0.5]
            # print(f"[Debug QSearch] Player {player} generated {len(q_moves)} quiescence moves: {q_moves[:5]}")
            return q_moves[:10] # 限制静态搜索分支因子
        else:
            # 普通搜索返回所有排序后的邻近点
            moves = [m for m, score, is_threat, is_defense in scored_potential_moves]
            # print(f"[Debug] Player {player} generated {len(moves)} moves (ordered): {moves[:10]}")
            # 限制分支广度 (可选，根据深度调整)
            # max_branch = 15 + (self.depth - current_depth_in_search) * 2
            # return moves[:max_branch]
            return moves


    def _quick_evaluate_move(self, board_arr, r, c, player):
        """快速评估落子 (r, c) 后 player 能形成的最高棋型分数"""
        opponent = PLAYER if player == AI else AI
        max_score = np.int64(0)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # 水平, 垂直, 主对角, 反对角

        for dr, dc in directions:
            line = []
            # 提取包含 (r, c) 的直线 (例如左右各取4格 + 中心点)
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    line.append(board_arr[nr, nc])
                else:
                    line.append(opponent) # 边界视作对手棋子

            # 评估这条线因为加入了 (r,c) 处的 player 棋子而产生的分数
            # 注意：这里 board_arr 已经在外部被修改了 (r,c) = player
            line_score = _evaluate_line_numba(np.array(line, dtype=np.int8), player, opponent)
            max_score = max(max_score, line_score)

        return max_score


    def _order_moves(self, board_arr, moves, player, h, depth, table):
        """
        对着法进行排序，提高剪枝效率
        - 置换表最佳着法
        - 启发式评估分数高的着法 (使用 _quick_evaluate_move)
        """
        opponent = PLAYER if player == AI else AI
        scored_moves = []

        # 1. 检查置换表是否有推荐的最佳着法
        tt_entry = table.get((h, depth))
        best_move_from_tt = tt_entry.get('best_move') if tt_entry else None

        # 2. 使用快速启发式评估给着法打分
        for r, c in moves:
            if board_arr[r, c] == EMPTY: # 再次确认
                 board_arr[r, c] = player
                 # 快速评估落子后的进攻和防守价值
                 score_p = self._quick_evaluate_move(board_arr, r, c, player)
                 board_arr[r, c] = EMPTY # 回溯

                 board_arr[r, c] = opponent
                 score_o = self._quick_evaluate_move(board_arr, r, c, opponent)
                 board_arr[r, c] = EMPTY # 回溯

                 # 启发式分数：己方得分 + 对手在此处得分 * 权重
                 heuristic_score = score_p + score_o * 1.5
                 scored_moves.append(((r, c), heuristic_score))

        # 3. 排序 (按启发式分数降序)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        ordered_moves = [m for m, s in scored_moves]

        # 4. 将置换表推荐的最佳着法放到最前面 (如果存在且有效)
        if best_move_from_tt and best_move_from_tt in ordered_moves:
            ordered_moves.remove(best_move_from_tt)
            ordered_moves.insert(0, best_move_from_tt)

        return ordered_moves


    def find_best_move(self):
        """主入口：寻找最佳着法 (使用多进程并行搜索第一层)"""
        self.start_time = time.time()
        self.nodes_searched = 0
        self.tt_hits = 0
        self.q_nodes_searched = 0
        local_tt = {} # 为这次搜索创建一个独立的根 TT (虽然子进程不用它)

        board_arr = self.initial_board_arr.copy()
        initial_hash = self._hash_board(board_arr)

        best_move = None
        best_score = -math.inf

        print(f"[AI] 开始搜索: 深度={self.depth}, 静态深度={self.quiescence_depth}, Workers={self.max_workers}, 棋盘={BOARD_SIZE}x{BOARD_SIZE}")

        # --- 生成并排序根节点的着法 ---
        moves = self._generate_moves(board_arr, AI, for_quiescence=False)
        if not moves: return self._fallback_move(board_arr) # 没有可走的路
        if len(moves) == 1:
             print(f"[AI] 唯一选择: {moves[0]}")
             return {"x": moves[0][1], "y": moves[0][0]} # 只有一个选择

        moves_ordered = self._order_moves(board_arr, moves, AI, initial_hash, self.depth, local_tt) # 根节点排序

        # 限制评估的顶级移动数量，以管理计算量 (特别是对于20x20)
        # 可以根据总移动数量动态调整，或者固定一个值
        max_root_moves_to_evaluate = min(len(moves_ordered), max(5, 4 + self.depth * 2)) # 例如：至少评估 5 个，深度越高评估越多
        moves_to_evaluate = moves_ordered[:max_root_moves_to_evaluate]
        print(f"[AI] 评估顶层 {len(moves_to_evaluate)} 个候选着法: {moves_to_evaluate}")


        # --- 使用进程池进行并行搜索 ---
        futures = {}
        results = {}
        all_tasks = []

        try:
            # 创建进程池
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                # 提交任务
                for r, c in moves_to_evaluate:
                    if board_arr[r, c] == EMPTY:
                        board_copy = board_arr.copy()
                        board_copy[r, c] = AI
                        h_new = self._update_hash(initial_hash, r, c, AI)
                        # 提交 Negamax 任务到进程池
                        # Negamax 需要从对手 (PLAYER) 的视角开始搜索下一层
                        future = pool.submit(self._negamax_process_wrapper,
                                             board_copy, self.depth - 1, -math.inf, math.inf, PLAYER, h_new, {}) # 子进程用空 TT
                        futures[future] = (r, c)
                        all_tasks.append(future)
                    else:
                         print(f"[AI] 警告: 考虑的着法 {(r, c)} 位置非空. 跳过.")

                # --- 获取结果 ---
                # 使用 wait 和 FIRST_COMPLETED 可以让我们在有结果时立即处理，
                # 而不是等待所有任务完成，这对于可能的超时处理更好。
                time_budget = self.time_limit_per_move * 0.95 # 留一点余量
                end_time = self.start_time + time_budget

                while futures: # 只要还有未完成的任务
                    # 等待至少一个任务完成，或者超时
                    done, not_done = wait(futures.keys(), timeout=max(0.1, end_time - time.time()), return_when=FIRST_COMPLETED)

                    current_time = time.time()
                    if not done and current_time >= end_time:
                         print("[AI] WARN: 超时！未能及时完成所有顶级评估。")
                         # 取消剩余任务 (如果 ProcessPoolExecutor 支持)
                         for f in not_done:
                              f.cancel() # 尝试取消
                         break # 跳出循环

                    for future in done:
                        move = futures.pop(future) # 从待处理中移除
                        try:
                            # 获取结果 (Negamax 返回的是对手视角的分数)
                            negamax_score = future.result()
                            # 转换回 AI 视角的分数
                            ai_score = -negamax_score
                            results[move] = ai_score
                            print(f"[AI]   评估 {move} -> 得分: {ai_score:.1f}")

                            # 提前更新最佳分数 (虽然最终会在所有结果出来后确认)
                            if ai_score > best_score:
                                best_score = ai_score
                                best_move = move

                        except TimeoutError: # 虽然 wait 超时了，但 result() 可能不抛 TimeoutError
                            print(f"[AI] Timeout (可能在 wait 之前) 评估着法 {move}. 赋予低分.")
                            results[move] = -math.inf - 100 # 超时惩罚
                        except Exception as e:
                            print(f"[AI] 错误 评估着法 {move}: {e}")
                            traceback.print_exc()
                            results[move] = -math.inf - 200 # 错误惩罚

                    if current_time >= end_time and futures:
                        print("[AI] WARN: 超时后跳出循环，可能有未完成的任务。")
                        break


            # --- 选择最佳着法 ---
            if results:
                 # 从已完成的评估中选择最佳着法
                 valid_results = {m: s for m, s in results.items() if s > -math.inf - 50} # 过滤掉惩罚分数

                 if valid_results:
                      best_move = max(valid_results, key=valid_results.get)
                      best_score = valid_results[best_move]
                 else: # 所有评估都超时或出错
                      print("[AI] WARN: 所有顶级评估超时或失败. 使用启发式回退.")
                      # Fallback: 从初始排序的移动中选择第一个
                      if moves_ordered:
                          best_move = moves_ordered[0]
                          # 计算其静态分数用于日志
                          board_arr[best_move[0], best_move[1]] = AI
                          best_score = self._evaluate_board(board_arr) # 静态评估
                          board_arr[best_move[0], best_move[1]] = EMPTY
                      else: # 不应该发生
                          fallback_coords = self._fallback_move(board_arr)
                          best_move = (fallback_coords['y'], fallback_coords['x'])
                          best_score = -math.inf

            else: # 如果没有收集到任何结果
                print("[AI] WARN: 未从工作进程收集到结果. 使用启发式回退.")
                if moves_ordered:
                    best_move = moves_ordered[0]
                    board_arr[best_move[0], best_move[1]] = AI
                    best_score = self._evaluate_board(board_arr)
                    board_arr[best_move[0], best_move[1]] = EMPTY
                else:
                    fallback_coords = self._fallback_move(board_arr)
                    best_move = (fallback_coords['y'], fallback_coords['x'])
                    best_score = -math.inf # Indicate fallback

        except Exception as e:
            print(f"[AI] CRITICAL 多进程池执行错误: {e}")
            traceback.print_exc()
            # 严重回退
            fallback_coords = self._fallback_move(board_arr)
            best_move = (fallback_coords['y'], fallback_coords['x'])
            best_score = -math.inf


        # --- 最终回退 ---
        if best_move is None:
            print("[AI] CRITICAL WARN: 搜索后未能确定最佳着法. 使用最终回退.")
            best_move_coords = self._fallback_move(board_arr)
            if "error" in best_move_coords: return best_move_coords # 传递错误
            best_move = (best_move_coords['y'], best_move_coords['x'])
            best_score = -math.inf

        elapsed_time = time.time() - self.start_time
        # 收集子进程统计信息 (如果可以的话)
        # total_nodes = self.nodes_searched + sum(fut.result()[1] for fut in completed_futures if fut.done() and not fut.cancelled())
        # total_q_nodes = self.q_nodes_searched + sum(...)
        # total_tt_hits = self.tt_hits + sum(...)
        # 简化: 只打印主进程看到的统计 (主要来自根节点评估)
        print(f"[AI] 选择着法: {best_move} 得分: {best_score:.1f} (耗时: {elapsed_time:.2f}s)")
        # print(f"[AI] 节点统计: 主搜索={self.nodes_searched}, 静态搜索={self.q_nodes_searched}, TT命中={self.tt_hits}")

        # 返回格式 {x, y}
        return {"x": best_move[1], "y": best_move[0]}


    def _fallback_move(self, board_arr, single_move=False):
        """备用走法：如果所有逻辑失败，随便找个空位"""
        print("[AI] 执行回退走法逻辑.")
        # 尝试中心点
        center_r, center_c = BOARD_SIZE // 2, BOARD_SIZE // 2
        if board_arr[center_r, center_c] == EMPTY:
            print("[AI] 回退: 中心点")
            move = (center_r, center_c)
            return move if single_move else {"x": center_c, "y": center_r}

        # 尝试有邻居的空点
        radius = 1
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_arr[r, c] == EMPTY:
                    has_neighbor_piece = False
                    for dr in range(-radius, radius + 1):
                       for dc in range(-radius, radius + 1):
                           if dr == 0 and dc == 0: continue
                           nr, nc = r + dr, c + dc
                           if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board_arr[nr, nc] != EMPTY:
                                has_neighbor_piece = True
                                break
                       if has_neighbor_piece: break
                    if has_neighbor_piece:
                         print(f"[AI] 回退: 第一个有邻居的空点 ({r}, {c})")
                         move = (r, c)
                         return move if single_move else {"x": c, "y": r}

        # 最后手段：随便找第一个空点
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_arr[r, c] == EMPTY:
                    print(f"[AI] 回退: 第一个找到的空点 ({r}, {c})")
                    move = (r, c)
                    return move if single_move else {"x": c, "y": r}

        print("[AI] 回退错误: 未找到空点!")
        return {"error": "棋盘已满或AI在回退中出错"}

    # --- Negamax with Alpha-Beta, Transposition Table, Quiescence Search ---
    # 注意：这个函数在子进程中执行，self 相关的状态 (nodes_searched 等) 不会直接累加到主进程
    def _negamax(self, board_arr, depth, alpha, beta, player, h, table):
        """Negamax 核心递归函数 (在子进程中运行)"""
        # self.nodes_searched += 1 # 无法直接修改主进程计数器
        node_count = 1
        q_node_count = 0
        tt_hit_count = 0

        original_alpha = alpha # 用于 TT 存储

        # 1. 检查置换表
        tt_key = (h, depth) # 使用哈希和深度作为键
        tt_entry = table.get(tt_key)
        if tt_entry:
            if tt_entry['depth'] >= depth: # 确保 TT 条目足够深或等于当前深度
                tt_hit_count += 1
                tt_score = tt_entry['score']
                tt_flag = tt_entry['flag']
                if tt_flag == 0: # Exact score (精确值)
                    # print(f"TT Exact Hit: depth={depth} score={tt_score}")
                    # 返回分数和统计信息
                    return tt_score, node_count, q_node_count, tt_hit_count
                elif tt_flag == 1: # Lower bound (之前在此节点发生了 beta 截断)
                    # print(f"TT Lower Hit: depth={depth} score>={tt_score}")
                    alpha = max(alpha, tt_score)
                elif tt_flag == 2: # Upper bound (之前在此节点 alpha 没有改善)
                    # print(f"TT Upper Hit: depth={depth} score<={tt_score}")
                    beta = min(beta, tt_score)

                if alpha >= beta:
                    # print(f"TT Cutoff: depth={depth} score={tt_score}")
                    # TT 信息导致剪枝
                    return tt_score, node_count, q_node_count, tt_hit_count

        # 2. 检查是否到达叶节点或终止状态
        # 使用评估函数检查，它内部包含胜利判断 (但效率可能不如直接检查)
        # 优化：先检查上一步是否导致胜负
        # (需要传递 last_move, Negamax 通常不直接传递，可在评估函数内部优化)
        static_eval = self._evaluate_board(board_arr) # 从 AI 视角评估
        current_player_eval = static_eval if player == AI else -static_eval # 转换为当前玩家视角

        # 如果分数绝对值很大，说明一方已理论获胜
        if abs(static_eval) >= SCORE_FIVE:
             # 深度加权，越快赢/输越好/差 (乘以剩余深度+1)
             return current_player_eval * (depth + 1), node_count, q_node_count, tt_hit_count

        # 到达主搜索深度，转入静态搜索
        if depth == 0:
            q_score, q_nodes, q_tt_hits = self._quiescence_search(board_arr, self.quiescence_depth, alpha, beta, player, h, table)
            q_node_count += q_nodes
            tt_hit_count += q_tt_hits
            return q_score, node_count, q_node_count, tt_hit_count

        # 3. 生成并排序着法
        moves = self._generate_moves(board_arr, player, for_quiescence=False)
        if not moves: # 没有可走的路 (平局或特殊情况)
             return 0, node_count, q_node_count, tt_hit_count
        # 在递归中对着法排序
        ordered_moves = self._order_moves(board_arr, moves, player, h, depth, table)

        # 4. 遍历子节点
        best_score = -math.inf
        best_move_for_tt = None # 记录导致最佳分数的移动

        opponent = PLAYER if player == AI else AI # 确定下一层的玩家

        for r, c in ordered_moves:
             if board_arr[r, c] == EMPTY: # 确保是空位
                board_arr[r, c] = player # 落子
                h_new = self._update_hash(h, r, c, player) # 更新哈希

                # 递归调用 Negamax，注意符号取反和 alpha/beta 交换
                # 返回值是 (score, nodes, q_nodes, tt_hits)
                child_score, child_nodes, child_q_nodes, child_tt_hits = self._negamax(
                    board_arr, depth - 1, -beta, -alpha, opponent, h_new, table
                )
                score = -child_score # Negamax 取反

                board_arr[r, c] = EMPTY # 回溯
                node_count += child_nodes
                q_node_count += child_q_nodes
                tt_hit_count += child_tt_hits

                # 更新 best_score 和 alpha
                if score > best_score:
                    best_score = score
                    best_move_for_tt = (r, c) # 记录最佳移动
                alpha = max(alpha, best_score)

                # Alpha-Beta 剪枝
                if alpha >= beta:
                    # print(f" Beta Cutoff: depth={depth} alpha={alpha} beta={beta}")
                    break # Beta cutoff

        # 5. 存储到置换表
        if best_score != -math.inf: # 只有找到有效移动才存储
            tt_entry_new = {'score': best_score, 'depth': depth, 'best_move': best_move_for_tt}
            if best_score <= original_alpha: # Failed low (未能改善 alpha), 是 Beta 节点 / Upper Bound
                tt_entry_new['flag'] = 2
            elif best_score >= beta: # Failed high (导致了 beta 截断), 是 Alpha 节点 / Lower Bound
                tt_entry_new['flag'] = 1
            else: # Exact score (在 alpha 和 beta 之间)
                tt_entry_new['flag'] = 0
            table[tt_key] = tt_entry_new
            # print(f"TT Store: key={(h % 1000, depth)} flag={tt_entry_new['flag']} score={best_score} move={best_move_for_tt}")

        # 返回分数和统计信息
        return best_score, node_count, q_node_count, tt_hit_count


    def _quiescence_search(self, board_arr, depth, alpha, beta, player, h, table):
        """静态威胁搜索 (只考虑威胁性和关键防守着法)"""
        # self.q_nodes_searched += 1 # 同样无法直接修改主进程计数器
        q_node_count = 1
        tt_hit_count = 0

        # 1. 评估当前局面 (stand pat score)
        static_eval = self._evaluate_board(board_arr)
        current_player_eval = static_eval if player == AI else -static_eval

        # 初始 alpha 更新：至少可以不走棋 (stand pat)
        alpha = max(alpha, current_player_eval)
        if alpha >= beta: # 如果当前局面已经比 beta 好，直接剪枝
            return beta, q_node_count, tt_hit_count # 返回 beta 作为下界
        if depth == 0: # 达到静态搜索深度限制
            return alpha, q_node_count, tt_hit_count # 返回当前最好的分数 alpha

        # 2. 只生成“战术性”着法 (如成四、成三、挡四、挡三等高价值点)
        moves = self._generate_moves(board_arr, player, for_quiescence=True)
        if not moves: # 没有威胁性走法，返回当前评估
            return alpha, q_node_count, tt_hit_count

        # 3. 遍历战术性着法 (也需要排序！)
        ordered_moves = self._order_moves(board_arr, moves, player, h, 0, table) # depth=0 for sorting

        opponent = PLAYER if player == AI else AI

        for r, c in ordered_moves:
            if board_arr[r, c] == EMPTY:
                board_arr[r, c] = player
                h_new = self._update_hash(h, r, c, player)

                score, child_q_nodes, child_tt_hits = self._quiescence_search(
                    board_arr, depth - 1, -beta, -alpha, opponent, h_new, table
                )
                score = -score # Negamax 取反

                board_arr[r, c] = EMPTY #回溯
                q_node_count += child_q_nodes
                tt_hit_count += child_tt_hits

                # 更新 alpha
                alpha = max(alpha, score)
                if alpha >= beta: # 剪枝
                    break

        return alpha, q_node_count, tt_hit_count


    def _negamax_process_wrapper(self, board_arr, depth, alpha, beta, player, h, table):
         """
         进程包装器，调用 Negamax 并返回结果。
         注意：此函数在独立的进程中执行。
         """
         try:
            # 调用 Negamax (返回 score, nodes, q_nodes, tt_hits)
            # 注意：这里的 table 是从父进程传来的空字典 {}
            result = self._negamax(board_arr, depth, alpha, beta, player, h, table)
            # 返回 Negamax 的原始分数 (相对于 player)
            # 统计信息可以在这里处理或传递，但为了简单，只返回分数
            return result[0] # 只返回分数
         except Exception as e:
             print(f"[工作进程错误] Negamax 执行异常: {e}")
             traceback.print_exc()
             # 返回一个极差的分数表示错误
             # 因为 Negamax 是从 player 角度，所以返回正无穷表示 player 输 (AI 赢)
             # 但我们需要包装器返回 AI 的视角，所以返回负无穷
             return -math.inf - 1000


# --- Flask 路由 ---
@app.route('/', methods=['GET'])
def index():
    # 使用 BASE_DIR 确保在打包后也能找到 index.html
    print(f"Serving index.html from: {BASE_DIR}")
    try:
        return send_from_directory(BASE_DIR, 'index.html')
    except Exception as e:
        print(f"Error serving index.html: {e}")
        # 尝试绝对路径（如果 BASE_DIR 不对）
        try:
            # 获取当前脚本所在的目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            return send_from_directory(script_dir, 'index.html')
        except Exception as e2:
             print(f"Error serving index.html from script dir: {e2}")
             return jsonify({"error": "Could not find index.html"}), 404


@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json
    if not data or 'board' not in data or 'depth' not in data:
        return jsonify({"error":"请求缺少 'board' 或 'depth' 参数"}), 400

    board_list = data['board']
    depth = int(data['depth'])
    # 'threads' 对应 'workers'
    workers = int(data.get('threads', os.cpu_count() or 1))

    # --- 输入验证 ---
    # 20x20 棋盘，深度限制需要更严格
    max_depth = 6 # 对于 20x20，深度 6 可能已经非常慢
    if not (1 <= depth <= max_depth):
        return jsonify({"error": f"搜索深度 (depth) 对于 20x20 棋盘必须在 1 到 {max_depth} 之间"}), 400
    if not (1 <= workers <= (os.cpu_count() or 1) * 2): # 限制 worker 数量
         return jsonify({"error": f"并行工作单元 (threads) 必须在 1 到 {(os.cpu_count() or 1) * 2} 之间"}), 400

    if depth > 4: # 深度大于 4 时给出警告
         print(f"WARN: 请求的深度 {depth} 在 {BOARD_SIZE}x{BOARD_SIZE} 棋盘上可能会非常慢!")

    # 验证棋盘结构和大小
    try:
        if not (isinstance(board_list, list) and len(board_list) == BOARD_SIZE and
                all(isinstance(row, list) and len(row) == BOARD_SIZE for row in board_list) and
                all(cell in [int(EMPTY), int(PLAYER), int(AI)] for row in board_list for cell in row)):
            raise ValueError(f"无效的棋盘结构或棋子值 (期望 {BOARD_SIZE}x{BOARD_SIZE})")
    except ValueError as e:
         return jsonify({"error": f"棋盘格式无效: {e}"}), 400

    # --- 创建 AI 实例并获取移动 ---
    print(f"收到请求: depth={depth}, workers={workers} (Numba {'可用' if numba else '不可用'}, 棋盘大小: {BOARD_SIZE}x{BOARD_SIZE})")
    ai = GomokuAI(board_list, depth, workers)
    try:
        move_result = ai.find_best_move()

        # 处理 AI 返回的潜在错误
        if isinstance(move_result, dict) and "error" in move_result:
            print(f"AI 错误: {move_result['error']}")
            return jsonify({"error": move_result['error']}), 500
        elif not isinstance(move_result, dict) or "x" not in move_result or "y" not in move_result:
            print("错误: AI 返回了无效的移动格式.")
            # 尝试最终回退
            fallback = ai._fallback_move(np.array(board_list, dtype=np.int8))
            if "error" in fallback:
                 return jsonify({"error": "AI 严重失败且回退失败."}), 500
            else:
                 return jsonify({"move": fallback, "warning": "AI 失败, 使用了绝对回退"})

        print(f"发送移动: {move_result}")
        # 可以在这里加入棋盘状态的日志记录
        # print_board(ai.initial_board_arr)
        return jsonify({"move": move_result})

    except Exception as e:
        print("="*20 + " AI 计算错误 " + "="*20)
        print(f"请求数据: depth={depth}, workers={workers}, size={BOARD_SIZE}")
        traceback.print_exc()
        print("="*50)
        return jsonify({"error": f"AI 内部计算错误，请检查服务器日志。"}), 500


# --- 启动与预编译 ---
def precompile_numba_functions():
    """尝试预编译 Numba 函数以减少首次运行延迟"""
    if not numba: return # 如果没有安装 Numba 则跳过
    try:
        print("正在预编译 Numba 函数 (可能需要一点时间)...")
        # 创建虚拟数据
        dummy_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        dummy_line = np.zeros(BOARD_SIZE, dtype=np.int8)
        dummy_board[BOARD_SIZE//2, BOARD_SIZE//2] = PLAYER

        # 调用需要编译的函数
        _check_win_numba(dummy_board, PLAYER, BOARD_SIZE//2, BOARD_SIZE//2, BOARD_SIZE)
        _evaluate_line_numba(dummy_line, PLAYER, AI)
        _calculate_player_score_numba(dummy_board, PLAYER, AI, BOARD_SIZE)

        # 也可以预热 AI 类中的 Numba 相关方法（如果它们被直接调用）
        # 例如评估函数和快速评估函数
        ai_instance = GomokuAI(dummy_board.tolist(), 1, 1)
        ai_instance._evaluate_board(dummy_board)
        # 放置一个棋子再快速评估
        dummy_board[BOARD_SIZE//2 + 1, BOARD_SIZE//2] = AI
        ai_instance._quick_evaluate_move(dummy_board, BOARD_SIZE//2 + 1, BOARD_SIZE//2, AI)

        print("Numba 预编译完成.")
    except Exception as e:
        print(f"Numba 预编译失败 (将在首次使用时编译): {e}")

# --- 主程序入口 ---
if __name__=='__main__':
    # 在 Windows 上使用 PyInstaller 打包时，需要这个
    # 但如果直接运行脚本，不需要，且应在主脚本调用
    # from multiprocessing import freeze_support
    # freeze_support()

    port = 5000
    print(f"=== 增强版五子棋 AI 服务器 (多进程 + Numba) 启动在端口 {port} ===")
    print(f"棋盘大小: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"检测到 CPU 核心数: {os.cpu_count() or '未知'}")
    print(f"Numba 是否可用: {'是' if numba else '否'}")

    # 尝试预编译
    precompile_numba_functions()

    # 运行 Flask 服务器
    # 使用 threaded=False 和 use_reloader=False 以避免与 ProcessPoolExecutor 冲突
    # debug=False 用于生产环境
    try:
        app.run(host='127.0.0.1', port=port, debug=False, threaded=False, use_reloader=False)
    except OSError as e:
         if "address already in use" in str(e):
              print(f"错误: 端口 {port} 已被占用。请关闭使用该端口的其他程序或更改端口号。")
         else:
              print(f"启动 Flask 服务器时发生 OS 错误: {e}")
    except Exception as e:
        print(f"启动 Flask 服务器时发生未知错误: {e}")