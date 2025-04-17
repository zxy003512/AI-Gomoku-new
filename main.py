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
    multiprocessing.freeze_support() # <--- 在这里添加 freeze_support()

    print("== AI 五子棋 启动中，请稍候 ==")
    # 启动 Flask 服务器的线程
    # daemon=True 意味着主线程退出时，这个服务器线程也会被强制退出
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 等待后端启动 (稍微优化一下等待逻辑)
    server_ready = False
    for i in range(20): # 最多等待 20 秒
        try:
            import urllib.request
            # 尝试访问一个简单的 health check 端点会更好，但这里直接访问根路径
            with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=1) as response:
                if response.status == 200:
                    server_ready = True
                    print("服务器已启动。")
                    break
        except Exception as e:
            # 打印更详细的等待信息，但不一定是错误
            # print(f"等待服务器启动... {i+1}/20 ({e})")
            print(f"等待服务器启动... {i+1}/20")
            time.sleep(1)

    if server_ready:
        print("打开浏览器界面...")
        webbrowser.open("http://127.0.0.1:5000")
    else:
        print("错误：服务器未能在预期时间内启动。请检查控制台输出。")
        # 可以在这里添加退出代码 sys.exit(1)

    # 保持主线程运行，以便服务器线程继续工作
    # 如果服务器线程是 daemon，主线程退出它也会退出
    # 如果希望手动停止（例如 Ctrl+C），可以保持主线程活动
    try:
        # 可以保持运行，或者等待服务器线程结束（如果它不是daemon）
        # server_thread.join() # 如果 server_thread 不是 daemon
        while True: # 保持主线程活动，允许通过 Ctrl+C 退出
             time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到退出信号，正在关闭...")
        # 这里可以添加一些清理代码（如果需要）
        print("程序退出。")
    except Exception as e:
        print(f"主线程发生意外错误: {e}")
        # 这里也可以添加清理代码