# -*- coding: utf-8 -*-
import threading, time, webbrowser, sys, os
from flask import Flask
import multiprocessing # <--- 导入 multiprocessing

def run_server():
    # 启动刚才写好的后端
    from gomoku_ai_server import app
    # 保证不使用 debug 模式
    # 注意：多进程模式下，不建议使用 Flask 开发服务器的 reloader
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False) # 明确禁用 reloader

if __name__=='__main__':
    # Crucial for multiprocessing when frozen (e.g., with PyInstaller)
    multiprocessing.freeze_support() # <--- Ensure this is called

    print("== AI 五子棋 启动中，请稍候 ==")
    # 启动 Flask 服务器的线程
    # daemon=True 意味着主线程退出时，这个服务器线程也会被强制退出
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 等待后端启动 (稍微优化一下等待逻辑)
    server_ready = False
    print("等待服务器响应 (最多 20 秒)...")
    for i in range(20): # 最多等待 20 秒
        try:
            import urllib.request
            # Try to access the root path
            with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=1) as response:
                if response.status == 200:
                    # Check if the server is printing the Numba message
                    # This is an indirect way, better check might be needed
                    server_ready = True
                    print(f"服务器在尝试 {i+1}/20 后响应。")
                    break
        except Exception as e:
            # Server might not be ready yet
            # print(f"等待服务器启动... {i+1}/20 ({type(e).__name__})")
            pass # Keep trying
        time.sleep(1)
        print(f"  尝试 {i+1}/20...")


    if server_ready:
        print("服务器已启动。打开浏览器界面...")
        webbrowser.open("http://127.0.0.1:5000")
    else:
        print("错误：服务器未能在预期时间内启动或响应。请检查控制台输出。")
        # Consider exiting if the server fails to start
        # sys.exit(1)

    # 保持主线程运行
    try:
        while True:
             time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到退出信号，正在关闭...")
        print("程序退出。")
    except Exception as e:
        print(f"主线程发生意外错误: {e}")