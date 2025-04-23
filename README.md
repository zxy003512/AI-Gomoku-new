# 🤖 优化版五子棋 AI 服务器

基于 Flask 和 Numba 的高性能五子棋 AI 服务器。

该项目实现了一个带有 Zobrist 哈希、Alpha-Beta 剪枝和 Numba 加速评估的五子棋 AI。通过提供一个简单的 HTTP API，它允许外部客户端（如配套的 HTML 页面）请求 AI 的最佳落子。

配套的前端界面 (`index.html`) 提供了一个可视化棋盘，可以与 AI 对战，或者进行 AI 之间的自动对弈，并调整 AI 的搜索深度和并行工作单元。

## ✨ 特性

- **高性能 AI**: 采用 Alpha-Beta 搜索、Zobrist 哈希、Killer Move、History Heuristic 等多种博弈树搜索优化技术。
- **Numba 加速**: 利用 Numba JIT 编译，大幅提升棋型评估和走法生成等核心算法的计算速度。
- **多进程/多线程**: 服务器后端 Flask 配置多线程/进程，AI 搜索层面支持并行工作单元，提高应对复杂局面的效率。
- **简单 API**: 提供 `/ai_move` HTTP POST 接口，接收当前棋盘状态和 AI 参数，返回计算出的最佳落子坐标。
- **配套前端**: 包含一个简洁的 HTML 界面，可以直接在浏览器中运行，提供人机对战和 AI 对战模式。
- **易于打包**: 可以通过 PyInstaller 打包成一个独立的 Windows/Linux/macOS 可执行文件。

！！由AI编写的五子棋，个人觉得还不错，陆陆续续也优化过不少
