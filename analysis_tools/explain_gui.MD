# gui.py 代码说明（LLM-Friendly）

## 1. 目标与功能概览

该脚本实现一个 **PyQt5 + pyqtgraph** 的实时轨迹监视器，用于显示来自 **C++ 端通过 UDP 发送的多轨迹位姿信息**，并提供交互控制与可视化工具：

- **三条轨迹显示**：`ESKF`、`DR`、`GNSS`
- **可切换渲染模式**：轨迹以 **Line / Points / Line+Points** 绘制
- **车辆姿态显示**：显示车体三角形（箭头）表示朝向；当选择 **GNSS** 作为 vehicle source 时不画箭头（无 heading）
- **轨迹角度对齐**：对每条轨迹提供独立的 **角度 offset（deg）**，绕轨迹起点旋转以进行对齐
- **视图模式**：Auto Follow（只平移不缩放）、Fit All、Fixed Window
- **历史回放**：滑条 scrub（拖动查看历史）
- **测距工具**：点击两点显示距离，并保留点与线；鼠标坐标常开显示
- **控制 C++ 程序**：Pause/Resume/Quit 通过 UDP 发送控制命令

---

## 2. 数据协议（UDP Telemetry）

### 2.1 Python 监听端口
- `TELEMETRY_PORT = 19001`：接收 C++ 发来的遥测包（JSON）

### 2.2 JSON 格式（约定）
每帧 JSON 包含：

- 顶层字段：
  - `t`: `float` 时间戳
  - `paused`: `int` 0/1
- 轨迹字段（可能存在或缺失）：
  - `eskf`, `dr`, `gnss`：每个为 dict，典型字段：
    - `x`: `float`
    - `y`: `float`
    - `yaw`: `float`（弧度）
    - `init`: `int` 0/1

Python 会将每条轨迹转换为统一内部结构并存储：  
`(t, x, y, yaw_deg, init, paused)`

---

## 3. 控制协议（Python -> C++）

- `CONTROL_PORT = 19002`：Python 向 C++ 发送控制消息
- JSON 格式示例：
  - `{"cmd":"pause"}`
  - `{"cmd":"resume"}`
  - `{"cmd":"quit"}`

发送入口函数：`send_cmd(cmd: str)`

---

## 4. 核心数据结构

### 4.1 轨迹历史缓存
```python
self.states = {
  "eskf": [ (t,x,y,yaw_deg,init,paused), ... ],
  "dr":   [ ... ],
  "gnss": [ ... ]
}
