# 五子棋 AI 服务器 (Gomoku AI Server - Python/Flask)

这是一个基于 Python 和 Flask 框架构建的五子棋 AI 后端服务器。它实现了一个使用 Minimax 算法（结合 Alpha-Beta 剪枝、Zobrist 哈希置换表、启发式评估和多线程优化）的 AI 对手，并提供了一个 API 接口供前端调用。

本项目还包含一个简单的 HTML/JavaScript 前端 (`index.html`) 用于演示和与 AI 对战。

## 主要功能

*   基于 Flask 的 Web 服务器。
*   使用 Minimax 算法及多种优化技术的 AI 引擎。
*   可配置的搜索深度（对应 AI 难度）。
*   通过 HTTP POST 请求提供 AI 下一步走法的 API。
*   启用 CORS，方便本地前后端分离开发。
*   包含一个基础的前端界面 (`index.html`)。

## 环境要求

*   Python (建议 3.8 或更高版本)
*   pip (Python 包管理器)
*   Web 浏览器 (用于运行 `index.html`)

## 安装与配置

1.  **获取代码:**
    克隆仓库或下载代码压缩包并解压。
    ```bash
    git clone <your-repo-url> # 或者直接下载解压
    ```

2.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖:**
    项目所需的 Python 库在 `requirements.txt` 文件中定义。运行以下命令安装：
    ```bash
    pip install -r requirements.txt
    ```
    主要依赖包括 `flask` 和 `flask-cors`。

## **使用方法 (重点)**

按照以下步骤启动并使用五子棋 AI 服务器和前端界面：

1.  **启动后端服务器:**
    *   确保你已经完成了上述“安装与配置”步骤，并且处于激活的虚拟环境中（如果使用了虚拟环境）。
    *   在项目根目录下，打开终端或命令行窗口。
    *   运行 Python 服务器脚本：
        ```bash
        python gomoku_ai_server.py
        ```
    *   启动成功后，你会看到类似以下的输出，表明服务器正在运行，并监听本地的 5000 端口：
        ```
         * Serving Flask app 'gomoku_ai_server'
         * Debug mode: on
         * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
         * Restarting with stat
         * Debugger is active!
         * Debugger PIN: xxx-xxx-xxx
        ```
        **注意:** 服务器默认运行在 `http://127.0.0.1:5000`，并且开启了调试模式（这在本地开发时很有用）。

2.  **打开前端游戏界面:**
    *   在你的文件管理器中，找到项目中的 `index.html` 文件。
    *   直接用你的 Web 浏览器（如 Chrome, Firefox, Edge 等）打开这个 `index.html` 文件。
        *   可以通过双击文件打开。
        *   或者在浏览器地址栏输入文件的本地路径 (例如 `file:///path/to/your/project/index.html`)。
    *   `index.html` 文件中的 JavaScript 代码已经配置为向 `http://127.0.0.1:5000/ai_move` 发送请求，也就是我们刚刚启动的本地服务器。

3.  **开始游戏:**
    *   **选择难度:** 在页面顶部的难度选择器中，点击选择你想要的 AI 难度（简单、普通、困难、极难）。这会设定 AI 思考的深度（`depth` 参数），选择后游戏会自动重置。默认是“普通”（深度 3）。
    *   **玩家下棋:** 默认玩家先手（执黑棋）。轮到你时，在棋盘的空白交叉点上点击鼠标左键即可落子。
    *   **AI 下棋:** 当你落子后，前端会将当前的棋盘状态和选择的难度（深度）发送给后端服务器 (`/ai_move` API)。服务器上的 AI 会进行计算。状态栏会显示“AI 思考中...🤔”。
    *   **AI 响应:** AI 计算完成后，服务器会返回 AI 决定下的位置。前端接收到响应后，会在棋盘上绘制 AI 的棋子（红棋）。
    *   **轮流进行:** 之后轮到玩家下棋，如此反复，直到一方获胜或棋盘下满（平局）。
    *   **状态显示:** 页面下方的状态栏会提示当前是谁的回合，或者显示游戏结果（“恭喜你赢了！🎉”、“AI赢了！🤖”、“平局！🤝”）。
    *   **控制按钮:**
        *   `重新开始`: 清空棋盘，重置游戏状态，玩家先手。
        *   `AI先手`: 清空棋盘，重置游戏状态，然后由 AI 首先进行思考并下第一步棋。

## API 端点说明

服务器提供以下 API 端点：

*   **URL:** `/ai_move`
*   **Method:** `POST`
*   **Request Body (JSON):**
    ```json
    {
      "board": [
        [0, 0, ..., 0],
        [0, 0, ..., 0],
        ... (15行)
        [0, 0, ..., 1, 2, 0, ...] // 15x15 的二维列表
      ],
      "depth": 3 // 整数，表示 AI 搜索深度
    }
    ```
    *   `board`: 一个 15x15 的嵌套列表，代表当前棋盘状态。`0` 表示空位，`1` 表示玩家（通常是先手，黑棋），`2` 表示 AI（通常是后手，红棋）。
    *   `depth`: 一个整数，指定 AI 使用 Minimax 算法搜索的深度。值越高，AI 越强，但计算时间也越长。

*   **Success Response (JSON, Status Code 200):**
    ```json
    {
      "move": {
        "y": 7, // AI 落子的行索引 (0-14)
        "x": 8  // AI 落子的列索引 (0-14)
      }
    }
    ```

*   **Error Response (JSON, Status Code 4xx or 5xx):**
    ```json
    {
      "error": "错误描述信息" // 例如 "Invalid board format", "Missing board or depth", "AI failed to find a move" 等
    }
    ```

## 注意事项

*   `index.html` 中的 `AI_BACKEND_URL` 变量当前设置为 `'http://127.0.0.1:5000/ai_move'`，仅适用于本地测试。如果将后端部署到其他地址，需要修改此 URL。
*   AI 的性能（思考速度）与选择的 `depth` 以及运行服务器的计算机性能密切相关。较高的深度（如 4 或 5）可能会导致明显的等待时间。
*   服务器代码 (`gomoku_ai_server.py`) 中开启了 Flask 的 `debug=True` 模式，这在开发时很有用，但在生产部署时应关闭。
*   代码中使用了多线程来加速顶层走法的评估，但限制了最大工作线程数（默认为 4），以避免在资源受限的环境（如某些免费云平台）中消耗过多资源。

