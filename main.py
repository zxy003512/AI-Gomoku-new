# -*- coding: utf-8 -*-
import threading, time, webbrowser, sys, os
from flask import Flask

def run_server():
    # 启动刚才写好的后端
    from gomoku_ai_server import app
    # 保证不使用 debug 模式
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__=='__main__':
    print("== AI 五子棋 启动中，请稍候 ==")
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    # 等待后端启动
    for i in range(20):
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:5000", timeout=1):
                break
        except Exception:
            print(f"等待服务器启动... {i+1}/20")
            time.sleep(1)
    print("打开浏览器界面...")
    webbrowser.open("http://127.0.0.1:5000")
    # 保持主线程不退出
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("退出程序。")
