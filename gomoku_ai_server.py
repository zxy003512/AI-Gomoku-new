# optimized_gomoku_ai_server.py
# -*- coding: utf-8 -*-
import os, sys, time, traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import numba
from numba import types, prange
from numba.typed import List

# --- 常量与配置 ---
SIZE = 20
BOARD_SIZE = SIZE
PLAYER = np.int8(1)
AI     = np.int8(2)
EMPTY  = np.int8(0)

# 棋型评分常量（和原版相同）
SCORE_FIVE            = 1000000000
SCORE_LIVE_FOUR       = 50000000
SCORE_RUSH_FOUR       = 4000000
SCORE_LIVE_THREE      = 3000000
SCORE_SLEEP_THREE     = 200000
SCORE_LIVE_TWO        = 5000
SCORE_SLEEP_TWO       = 300
SCORE_LIVE_ONE        = 10
SCORE_SLEEP_ONE       = 2

# Zobrist 哈希
zobrist_table = np.random.randint(
    np.iinfo(np.uint64).min, np.iinfo(np.uint64).max,
    size=(BOARD_SIZE,BOARD_SIZE,3), dtype=np.uint64
)

# ---------------------------------------------------------------------
# Numba 加速：连五检测 / 单行评估 / 全盘评估（与原版完全相同，略去解释）
# ---------------------------------------------------------------------
@numba.njit("b1(int8[:,:], i8, i8, i8, i4)", cache=True, fastmath=True)
def _check_win_numba(board, player, x, y, bs):
    if not (0 <= x < bs and 0 <= y < bs and board[y,x] == player):
        return False
    dirs = ((1,0),(0,1),(1,1),(1,-1))
    for dx,dy in dirs:
        cnt = 1
        for sign in (-1,1):
            for i in range(1,5):
                nx = x + dx*sign*i; ny = y + dy*sign*i
                if nx<0 or ny<0 or nx>=bs or ny>=bs or board[ny,nx]!=player:
                    break
                cnt += 1
        if cnt>=5:
            return True
    return False

@numba.njit(cache=True, fastmath=True)
def _evaluate_line_numba(line, player, opponent):
    # （此处直接原封不动沿用你的完整版本）
    score = np.int64(0)
    n = line.shape[0]
    if n < 5: return score
    empty = EMPTY
    live_four = rush_four = 0
    live_three = sleep_three = 0
    live_two = sleep_two = 0
    live_one = sleep_one = 0

    # 窗口 5 检冲四、眠三
    for i in range(n-4):
        w = line[i:i+5]
        p_cnt=o_cnt=0
        for v in w:
            if v==player: p_cnt+=1
            elif v==opponent: o_cnt+=1
        if o_cnt==0:
            if p_cnt==4:
                # ... 冲四判定略（沿用原版）
                left_empty = (i>0 and line[i-1]==empty)
                right_empty= (i+5<n and line[i+5]==empty)
                is_rush=False
                if w[0]==empty and (i+5==n or line[i+5]==opponent):
                    if left_empty: is_rush=True
                elif w[4]==empty and (i==0 or line[i-1]==opponent):
                    if right_empty: is_rush=True
                else:
                    if left_empty or right_empty: is_rush=True
                if is_rush: rush_four+=1
            elif p_cnt==3:
                # 眠三简化判定
                for start in range(3):
                    if w[start]==player and w[start+1]==player and w[start+2]==player:
                        leftb = (i==0 or line[i-1]==opponent)
                        rightb= (i+5==n or line[i+5]==opponent)
                        if leftb or rightb:
                            sleep_three+=1
                            break

    # 窗口 6 检活四、活三、活二
    for i in range(n-5):
        w = line[i:i+6]
        if w[0]==empty and w[5]==empty:
            p_cnt=o_cnt=0
            for v in w:
                if v==player: p_cnt+=1
                elif v==opponent: o_cnt+=1
            if o_cnt==0:
                if p_cnt==4:
                    live_four+=1
                elif p_cnt==3:
                    if (w[1]==player and w[2]==player and w[3]==player) or \
                       (w[2]==player and w[3]==player and w[4]==player) or \
                       (w[1]==player and w[3]==player and w[4]==player) or \
                       (w[1]==player and w[2]==player and w[4]==player):
                        live_three+=1
                elif p_cnt==2:
                    if (w[1]==player and w[2]==player) or \
                       (w[2]==player and w[3]==player) or \
                       (w[3]==player and w[4]==player) or \
                       (w[1]==player and w[3]==player) or \
                       (w[2]==player and w[4]==player) or \
                       (w[1]==player and w[4]==player):
                        live_two+=1

    # 眠二、眠一
    i=0
    while i<n:
        if line[i]==player:
            cnt=1; i2=i+1
            while i2<n and line[i2]==player:
                cnt+=1; i2+=1
            left_open = (i>0 and line[i-1]==empty)
            right_open= (i2<n and line[i2]==empty)
            opens = (1 if left_open else 0)+(1 if right_open else 0)
            if cnt==2 and opens==1:
                sleep_two+=1
            elif cnt==1 and opens==2:
                live_one+=1
            elif cnt==1 and opens==1:
                sleep_one+=1
            i = i2
        else:
            i+=1

    # 组合加分
    if live_four>0 or rush_four>=2:
        score += int(SCORE_FIVE*0.95)
    elif live_three>=2:
        score += int(SCORE_LIVE_FOUR*0.9)
    elif live_three>=1 and rush_four>=1:
        score += int(SCORE_LIVE_FOUR*0.95)

    score += live_four*SCORE_LIVE_FOUR
    score += rush_four*SCORE_RUSH_FOUR
    score += live_three*SCORE_LIVE_THREE
    score += sleep_three*(SCORE_SLEEP_THREE if live_four==0 and rush_four==0 else SCORE_SLEEP_THREE//2)
    score += live_two*(SCORE_LIVE_TWO if live_four==0 and rush_four==0 and live_three==0 else SCORE_LIVE_TWO//2)
    score += sleep_two*SCORE_SLEEP_TWO
    score += live_one*SCORE_LIVE_ONE
    score += sleep_one*SCORE_SLEEP_ONE

    if score > np.iinfo(np.int64).max:    score = np.iinfo(np.int64).max
    if score < np.iinfo(np.int64).min:    score = np.iinfo(np.int64).min
    return score

@numba.njit(cache=True, fastmath=True)
def _calculate_player_score_numba(board, player, opponent, bs):
    total = np.int64(0)
    # 行/列
    for i in range(bs):
        total += _evaluate_line_numba(board[i,:],   player,opponent)
        total += _evaluate_line_numba(board[:,i],   player,opponent)
    # 两斜
    for off in range(-(bs-5), bs-4):
        d1 = np.diag(board, k=off)
        if d1.shape[0]>=5: total += _evaluate_line_numba(d1, player,opponent)
        d2 = np.diag(np.fliplr(board), k=off)
        if d2.shape[0]>=5: total += _evaluate_line_numba(d2, player,opponent)
    return total

# ---------------------------------------------------------
# Numba 加速：快速局部评估（Inline _evaluate_line）和候选生成
# ---------------------------------------------------------
# 定义 List 的元素类型：两个 int64
pos_t = types.UniTuple(types.int64,2)

@numba.njit(cache=True)
def _quick_eval_numba(B, r, c, pl):
    # 只计算点 (r,c) 四个方向上的最大棋型分
    opp = PLAYER if pl==AI else AI
    maxs = np.int64(0)
    dirs = ((1,0),(0,1),(1,1),(1,-1))
    for d in dirs:
        # 拷贝长度 9 的 line
        tmp = np.empty(9, dtype=np.int8)
        idx = 0
        for i in range(-4,5):
            rr = r + d[0]*i; cc = c + d[1]*i
            if 0<=rr<BOARD_SIZE and 0<=cc<BOARD_SIZE:
                tmp[idx] = B[rr,cc]
            else:
                tmp[idx] = opp
            idx+=1
        sc = _evaluate_line_numba(tmp, pl, opp)
        if sc>maxs: maxs=sc
    return maxs

@numba.njit(cache=True)
def _generate_moves_numba(B, player, for_quiescence):
    bs = BOARD_SIZE
    empty = EMPTY
    opp   = PLAYER if player==AI else AI

    # 1) 一步赢
    imm = List.empty_list(pos_t)
    for r in range(bs):
        for c in range(bs):
            if B[r,c]==empty:
                B[r,c]=player
                if _check_win_numba(B, player, c, r, bs):
                    imm.append((r,c))
                B[r,c]=empty
    if len(imm)>0:
        return imm

    # 2) 必须堵
    blk = List.empty_list(pos_t)
    for r in range(bs):
        for c in range(bs):
            if B[r,c]==empty:
                B[r,c]=opp
                if _check_win_numba(B, opp, c, r, bs):
                    blk.append((r,c))
                B[r,c]=empty
    if len(blk)>0:
        return blk

    # 3) 邻近已有子的空点
    cand = List.empty_list(pos_t)
    for r in range(bs):
        for c in range(bs):
            if B[r,c]!=empty: continue
            nb = False
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    rr = r+dr; cc = c+dc
                    if dr==0 and dc==0: continue
                    if 0<=rr<bs and 0<=cc<bs and B[rr,cc]!=empty:
                        nb = True; break
                if nb: break
            if nb:
                cand.append((r,c))

    # 空盘直接中心
    if len(cand)==0:
        mid = bs//2
        out = List.empty_list(pos_t)
        out.append((mid,mid))
        return out

    # 4) quiescence 时只留高威胁
    if for_quiescence:
        qm = List.empty_list(pos_t)
        for idx in range(len(cand)):
            r,c = cand[idx]
            sc = _quick_eval_numba(B, r, c, player)
            if sc >= SCORE_SLEEP_THREE:
                qm.append((r,c))
                if len(qm)>=10: break
        return qm

    # 普通情况，返回完整列表，由 Python 侧再进行排序
    return cand

# ---------------------------------------------------------------------
# Python 层接口：只负责搜索框架，生成/排序走法时调用上面的 Numba 函数
# ---------------------------------------------------------------------
class GomokuAI:
    def __init__(self, board, max_depth, max_workers=None):
        self.board0 = np.array(board, dtype=np.int8)
        self.max_depth = max(1, max_depth)
        self.q_depth   = max(1, self.max_depth//2)
        self.time_limit= 8.0
        cpu = os.cpu_count() or 1
        self.workers  = max(1, min(max_workers or cpu, cpu, 8))
        self.TT       = {}
        self.killer   = [[(-1,-1),(-1,-1)] for _ in range(self.max_depth+2)]
        self.history  = {}
        self.nodes    = 0

    def _hash_board(self, B):
        h = np.uint64(0)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = B[r,c]
                if p!=EMPTY:
                    h ^= zobrist_table[r,c,int(p)]
        return h

    def _update_hash(self, h, r, c, p, old=EMPTY):
        if old!=EMPTY:
            h ^= zobrist_table[r,c,int(old)]
        if p!=EMPTY:
            h ^= zobrist_table[r,c,int(p)]
        return h

    def _evaluate(self, B, last_move=None):
        if last_move:
            rr,cc,pl = last_move
            if _check_win_numba(B,pl,cc,rr,BOARD_SIZE):
                return SCORE_FIVE*10 if pl==AI else -SCORE_FIVE*10
        ai_s = _calculate_player_score_numba(B, AI, PLAYER, BOARD_SIZE)
        pl_s = _calculate_player_score_numba(B, PLAYER, AI, BOARD_SIZE)
        if ai_s>=SCORE_FIVE: return float(SCORE_FIVE*10)
        if pl_s>=SCORE_FIVE: return float(-SCORE_FIVE*10)
        return float(ai_s - pl_s*1.1)

    def _generate_moves(self, B, player, for_q):
        # 调用 Numba 层
        return _generate_moves_numba(B, player, for_q)

    def _order_moves(self, B, moves, player, h, depth):
        # PV / killer 优先（同原版）
        key = (h, depth, player)
        ttent = self.TT.get(key, {})
        best_tt = ttent.get('best_move', None)
        k1,k2 = self.killer[self.max_depth-depth]
        hist = self.history

        # Python 层排序：调用 Numba 的快速局部评估
        scored = []
        opp = PLAYER if player==AI else AI
        for m in moves:
            r,c = m
            if m == best_tt:
                scored.append((m, 1e14)); continue
            if m == k1:
                scored.append((m, 1e13)); continue
            if m == k2:
                scored.append((m, 1e12)); continue
            # history
            hsc = hist.get(m,0)
            # Numba 快评
            ps = _quick_eval_numba(B, r, c, player)
            os = _quick_eval_numba(B, r, c, opp)
            sc = hsc + ps + os*0.3
            scored.append((m, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m,_ in scored]

    # _negamax, _quiescence, find_best_move 结构与原版完全相同，只是内部调用上面优化后的函数
    def _negamax(self, B, depth, alpha, beta, player, h, table):
        self.nodes += 1
        orig_alpha = alpha
        pv=[]
        # TT 检查
        key = (h,depth,player)
        ttent = self.TT.get(key)
        if ttent and ttent['depth']>=depth:
            sc,fl,bmv = ttent['score'], ttent['flag'], ttent['best_move']
            if fl==0: return sc,0,0,1,{'pv':[bmv],'cutoff':False}
            if fl==1: alpha=max(alpha,sc)
            if fl==2: beta=min(beta,sc)
            if alpha>=beta:
                return sc,0,0,1,{'pv':[bmv],'cutoff':True}
        # 叶节点或静评
        static = self._evaluate(B)
        cur = static if player==AI else -static
        if depth<=0 or abs(static)>=SCORE_FIVE:
            if depth<=0:
                qsc,_,_ = self._quiescence(B, self.q_depth, alpha, beta, player, h, table)
                return qsc,0,0,0,{'pv':[],'cutoff':False}
            return cur*(depth+1),0,0,0,{'pv':[],'cutoff':False}
        # Null‐Move
        if depth>=3:
            opp = PLAYER if player==AI else AI
            nm_sc,_,_,_,_ = self._negamax(B, depth-1-2, -beta, -alpha, opp, h, table)
            nm_sc = -nm_sc
            if nm_sc>=beta:
                return nm_sc,0,0,0,{'pv':[],'cutoff':True}
        # 生成 + 排序
        moves = self._generate_moves(B, player, False)
        if not moves: return 0,0,0,0,{'pv':[],'cutoff':False}
        ord_moves = self._order_moves(B, moves, player, h, depth)

        best_sc=-1e18; best_mv=None; first=True; opp = PLAYER if player==AI else AI
        for mv in ord_moves:
            r,c=mv
            B[r,c]=player
            h2 = self._update_hash(h,r,c,player)
            if first:
                sc,_,_,_,sub = self._negamax(B, depth-1, -beta, -alpha, opp, h2, table)
                sc = -sc; first=False
            else:
                sc,_,_,_,sub = self._negamax(B, depth-1, -alpha-1, -alpha, opp, h2, table)
                sc = -sc
                if alpha<sc<beta:
                    sc,_,_,_,sub = self._negamax(B, depth-1, -beta, -sc, opp, h2, table)
                    sc = -sc
            B[r,c]=EMPTY
            if sc>best_sc:
                best_sc=sc; best_mv=mv; pv=[mv]+sub.get('pv',[])
            alpha = max(alpha, sc)
            if alpha>=beta:
                # killer 更新
                idx = self.max_depth-depth
                if best_mv not in self.killer[idx]:
                    self.killer[idx][1]=self.killer[idx][0]
                    self.killer[idx][0]=best_mv
                break

        # history & TT 存储
        if best_mv and alpha<beta:
            self.history[best_mv] = self.history.get(best_mv,0) + depth*depth
        flag = 0
        if best_sc<=orig_alpha: flag=2
        elif best_sc>=beta: flag=1
        self.TT[key]={'score':best_sc,'flag':flag,'depth':depth,'best_move':best_mv}

        return best_sc,0,0,0,{'pv':pv,'cutoff':False}

    def _quiescence(self, B, depth, alpha, beta, player, h, table):
        stand = self._evaluate(B)
        cur   = stand if player==AI else -stand
        alpha = max(alpha, cur)
        if depth<=0 or alpha>=beta:
            return alpha,0,0
        moves = self._generate_moves(B, player, True)
        best = alpha; opp = PLAYER if player==AI else AI
        for mv in moves:
            r,c = mv
            B[r,c]=player
            h2 = self._update_hash(h,r,c,player)
            sc,_,_ = self._quiescence(B, depth-1, -beta, -alpha, opp, h2, table)
            sc = -sc
            B[r,c]=EMPTY
            if sc>best: best=sc
            alpha = max(alpha, sc)
            if alpha>=beta:
                break
        return alpha,0,0

    def find_best_move(self):
        start = time.time()
        B0 = self.board0.copy()
        h0 = self._hash_board(B0)
        best_move=None; prev=0.0; window=SCORE_FIVE
        for d in range(1, self.max_depth+1):
            alpha = prev-window; beta = prev+window
            sc,_,_,_,info = self._negamax(B0, d, alpha, beta, AI, h0, self.TT)
            if info['cutoff']:
                sc,_,_,_,info = self._negamax(B0, d, -1e15, 1e15, AI, h0, self.TT)
            prev = sc
            if time.time()-start>self.time_limit:
                break
            if info['pv']:
                best_move = info['pv'][0]
        if not best_move:
            mvs = self._generate_moves(B0, AI, False)
            best_move = mvs[0] if mvs else (BOARD_SIZE//2,BOARD_SIZE//2)
        return {"x": best_move[1], "y": best_move[0]}

# ---------------------------------------------------------------------
# Flask 路由
# ---------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
CORS(app)
BASE_DIR = getattr(sys,'_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

@app.route('/', methods=['GET'])
def index():
    try:
        return send_from_directory(BASE_DIR, 'index.html')
    except:
        return "<h3>index.html 未找到</h3>", 404

@app.route('/ai_move', methods=['POST'])
def ai_move():
    try:
        data = request.json or {}
        board = data.get('board')
        depth = int(data.get('depth', 3))
        tw    = int(data.get('threads', os.cpu_count() or 1))
        if not board or len(board)!=BOARD_SIZE or any(len(r)!=BOARD_SIZE for r in board):
            return jsonify(error="无效棋盘"), 400
        depth = max(1, min(depth, 8))
        ai = GomokuAI(board, depth, tw)
        mv = ai.find_best_move()
        return jsonify(move=mv)
    except:
        traceback.print_exc()
        return jsonify(error="AI 计算出错"), 500

def precompile():
    # 提前触发 Numba 编译
    dummy = np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.int8)
    dummy[BOARD_SIZE//2,BOARD_SIZE//2] = PLAYER
    _check_win_numba(dummy, PLAYER, BOARD_SIZE//2, BOARD_SIZE//2, BOARD_SIZE)
    _calculate_player_score_numba(dummy, PLAYER, AI, BOARD_SIZE)
    line = np.zeros(BOARD_SIZE, dtype=np.int8)
    _evaluate_line_numba(line, PLAYER, AI)
    _quick_eval_numba(dummy, BOARD_SIZE//2, BOARD_SIZE//2, PLAYER)
    _generate_moves_numba(dummy, PLAYER, False)
    _generate_moves_numba(dummy, PLAYER, True)

if __name__=='__main__':
    precompile()
    print(f"=== 优化版 五子棋 AI 服务器 启动 on 127.0.0.1:5000 (Board {BOARD_SIZE}x{BOARD_SIZE}) ===")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=False, use_reloader=False)
