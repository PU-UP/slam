import sys, json, socket, math, time
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

TELEMETRY_PORT = 19001  # C++ -> Python
CONTROL_PORT   = 19002  # Python -> C++
CONTROL_IP     = "127.0.0.1"

def wrap_deg(deg: float) -> float:
    while deg > 180: deg -= 360
    while deg < -180: deg += 360
    return deg

def rot2d(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trajectory + Vehicle Heading Monitor")

        # UDP RX (telemetry)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", TELEMETRY_PORT))
        self.rx.setblocking(False)

        # UDP TX (control)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # === 历史缓存：每个点包含 (t, x, y, yaw_deg, init, paused) ===
        self.states = []
        self.last_t = None

        # Scrub / live control
        self.live = True          # True: 显示最新；False: 显示 slider 指定的历史
        self.scrubbing = False    # slider 正在被拖动
        self.selected_idx = -1    # 当前显示的 index

        # rate stats
        self.msg_count = 0
        self.last_rate_ts = time.time()

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "x")
        self.plot.setLabel("left", "y")

        # 轨迹：明确画线（避免 pen=None 看不见）
        self.curve = self.plot.plot([], [], pen=pg.mkPen(width=2), name="trajectory")

        # 当前中心点
        self.pos_scatter = pg.ScatterPlotItem(size=8, pen=None)
        self.plot.addItem(self.pos_scatter)

        # 车体三角形（尖端指向航向）
        self.vehicle_item = pg.PlotDataItem([], [], pen=None)
        self.plot.addItem(self.vehicle_item)

        # 车体形状（局部坐标）：默认车头朝 +X
        self.vehicle_scale = 1.0  # 想更小改 0.6，想更大改 1.5/2.0
        self.vehicle_shape_local = np.array([
            [ 1.0,  0.0],   # 车头尖
            [-0.7,  0.5],   # 左后
            [-0.7, -0.5],   # 右后
            [ 1.0,  0.0],   # 闭合
        ], dtype=float) * self.vehicle_scale

        # View mode
        self.view_mode = "follow"   # follow | fit | fixed
        self.fixed_window = 40.0    # 坐标单位窗口边长（跟随/固定模式使用）

        # Right panel labels
        self.yaw_label = QtWidgets.QLabel("yaw: --- deg")
        self.yaw_label.setStyleSheet("font-size: 24px; font-weight: 600;")
        self.state_label = QtWidgets.QLabel("state: ---")
        self.t_label = QtWidgets.QLabel("t: ---")
        self.pos_label = QtWidgets.QLabel("pos: ---")
        self.rate_label = QtWidgets.QLabel("rx: --- Hz")
        self.mode_label = QtWidgets.QLabel("mode: LIVE")

        # Buttons
        btn_pause  = QtWidgets.QPushButton("Pause")
        btn_resume = QtWidgets.QPushButton("Resume")
        btn_quit   = QtWidgets.QPushButton("Quit")
        btn_clear  = QtWidgets.QPushButton("Clear Track")
        btn_back_now = QtWidgets.QPushButton("Back to Now")

        btn_pause.clicked.connect(lambda: self.send_cmd("pause"))
        btn_resume.clicked.connect(lambda: self.send_cmd("resume"))
        btn_quit.clicked.connect(lambda: self.send_cmd("quit"))
        btn_clear.clicked.connect(self.clear_track)
        btn_back_now.clicked.connect(self.back_to_now)

        # View controls
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Auto Follow", "Fit All", "Fixed Window"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(5.0, 500.0)
        self.window_spin.setSingleStep(5.0)
        self.window_spin.setValue(self.fixed_window)
        self.window_spin.valueChanged.connect(self.on_window_changed)

        # === Progress slider (scrub) ===
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setTracking(True)  # 拖动时实时更新
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)

        self.slider_info = QtWidgets.QLabel("0 / 0")

        # Layout
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.yaw_label)
        right.addWidget(self.mode_label)
        right.addWidget(self.state_label)
        right.addWidget(self.t_label)
        right.addWidget(self.pos_label)
        right.addWidget(self.rate_label)
        right.addSpacing(10)

        right.addWidget(QtWidgets.QLabel("Progress (scrub history)"))
        right.addWidget(self.slider)
        right.addWidget(self.slider_info)
        right.addWidget(btn_back_now)
        right.addSpacing(10)

        right.addWidget(QtWidgets.QLabel("View Mode"))
        right.addWidget(self.mode_combo)

        roww = QtWidgets.QHBoxLayout()
        roww.addWidget(QtWidgets.QLabel("Window"))
        roww.addWidget(self.window_spin)
        right.addLayout(roww)

        right.addSpacing(10)

        row_btn = QtWidgets.QHBoxLayout()
        row_btn.addWidget(btn_pause)
        row_btn.addWidget(btn_resume)
        row_btn.addWidget(btn_quit)
        right.addLayout(row_btn)

        right.addWidget(btn_clear)
        right.addStretch(1)

        main = QtWidgets.QHBoxLayout()
        main.addWidget(self.plot, stretch=3)
        main.addLayout(right, stretch=1)
        self.setLayout(main)

        # Timer: poll telemetry
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(20)  # 50Hz poll

        # Timer: update rate label
        self.rate_timer = QtCore.QTimer(self)
        self.rate_timer.timeout.connect(self.update_rate)
        self.rate_timer.start(500)

    # ===== Control =====
    def send_cmd(self, cmd: str):
        msg = json.dumps({"cmd": cmd}, separators=(",", ":")).encode("utf-8")
        self.tx.sendto(msg, (CONTROL_IP, CONTROL_PORT))

    # ===== View mode =====
    def on_mode_changed(self, idx: int):
        if idx == 0:
            self.view_mode = "follow"
        elif idx == 1:
            self.view_mode = "fit"
        else:
            self.view_mode = "fixed"

    def on_window_changed(self, v: float):
        self.fixed_window = float(v)

    # ===== Slider events =====
    def on_slider_pressed(self):
        self.scrubbing = True
        self.live = False
        self.mode_label.setText("mode: HISTORY")

    def on_slider_released(self):
        self.scrubbing = False
        # 释放后仍停留在 HISTORY，直到点 Back to Now

    def on_slider_value_changed(self, v: int):
        if len(self.states) == 0:
            return
        self.selected_idx = int(np.clip(v, 0, len(self.states) - 1))
        if not self.live:
            self.render_at_index(self.selected_idx)

    def back_to_now(self):
        if len(self.states) == 0:
            return
        self.live = True
        self.mode_label.setText("mode: LIVE")
        self.selected_idx = len(self.states) - 1
        # 把 slider 拉回末尾（不触发“暂停显示”，只是更新显示）
        self.slider.blockSignals(True)
        self.slider.setValue(self.selected_idx)
        self.slider.blockSignals(False)
        self.render_at_index(self.selected_idx)

    # ===== Data / rendering =====
    def clear_track(self):
        self.states.clear()
        self.last_t = None
        self.curve.setData([], [])
        self.pos_scatter.setData([])
        self.vehicle_item.setData([], [])
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider_info.setText("0 / 0")
        self.mode_label.setText("mode: LIVE")
        self.live = True
        self.selected_idx = -1

    def update_rate(self):
        now = time.time()
        dt = now - self.last_rate_ts
        hz = self.msg_count / dt if dt > 1e-6 else 0.0
        self.rate_label.setText(f"rx: {hz:.1f} Hz")
        self.msg_count = 0
        self.last_rate_ts = now

    def update_view(self, xs, ys, px: float, py: float):
        if self.view_mode == "follow":
            half = self.fixed_window * 0.5
            self.plot.setXRange(px - half, px + half, padding=0)
            self.plot.setYRange(py - half, py + half, padding=0)

        elif self.view_mode == "fit":
            if len(xs) >= 2:
                xmin, xmax = float(np.min(xs)), float(np.max(xs))
                ymin, ymax = float(np.min(ys)), float(np.max(ys))
                padx = max(1.0, 0.1 * (xmax - xmin))
                pady = max(1.0, 0.1 * (ymax - ymin))
                self.plot.setXRange(xmin - padx, xmax + padx, padding=0)
                self.plot.setYRange(ymin - pady, ymax + pady, padding=0)

        else:  # fixed
            half = self.fixed_window * 0.5
            self.plot.setXRange(px - half, px + half, padding=0)
            self.plot.setYRange(py - half, py + half, padding=0)

    def render_at_index(self, idx: int):
        """显示历史 idx：轨迹画到 idx，车体显示 idx pose。"""
        if len(self.states) == 0:
            return
        idx = int(np.clip(idx, 0, len(self.states) - 1))

        # 取到 idx 的轨迹
        arr = np.array(self.states[:idx+1], dtype=float)  # columns: t,x,y,yaw,init,paused
        xs = arr[:, 1]
        ys = arr[:, 2]

        # 当前 pose
        t, px, py, yaw_deg, init, paused = self.states[idx]

        # labels
        if int(init) == 0:
            self.yaw_label.setText("yaw: --- deg")
        else:
            self.yaw_label.setText(f"yaw: {wrap_deg(yaw_deg):+.2f} deg")
        self.state_label.setText(f"state: {'PAUSED' if int(paused) else 'RUN'} | init: {int(init)}")
        self.t_label.setText(f"t: {t:.3f}")
        self.pos_label.setText(f"pos: [{px:.3f}, {py:.3f}, 0.000]")  # z 不在历史里就先不显示

        # slider info
        self.slider_info.setText(f"{idx+1} / {len(self.states)}")

        # plot
        self.curve.setData(xs, ys)
        self.pos_scatter.setData([px], [py])

        # 车体三角形（yaw 不加负号：你要求的）
        theta = math.radians(yaw_deg)
        R = rot2d(theta)
        pts = (self.vehicle_shape_local @ R.T) + np.array([px, py], dtype=float)

        # 填充车体（RGBA）
        self.vehicle_item.setData(
            pts[:, 0], pts[:, 1],
            fillLevel=0,
            brush=pg.mkBrush(50, 150, 255, 120)
        )

        # view（跟随/固定看当前 pose；fit 看全轨迹）
        self.update_view(xs, ys, px, py)

    def poll(self):
        updated = False
        while True:
            try:
                data, _ = self.rx.recvfrom(8192)
            except BlockingIOError:
                break

            self.msg_count += 1
            try:
                obj = json.loads(data.decode("utf-8", errors="ignore"))
            except Exception:
                continue

            yaw_rad = obj.get("yaw", None)
            init = int(obj.get("init", 0))
            paused = int(obj.get("paused", 0))
            t = float(obj.get("t", 0.0))
            px = float(obj.get("px", 0.0))
            py = float(obj.get("py", 0.0))

            if yaw_rad is None or init == 0:
                yaw_deg = 0.0
            else:
                yaw_deg = wrap_deg(float(yaw_rad) * 180.0 / math.pi)

            # 去重：避免同时间戳重复堆积
            if self.last_t is None or abs(t - self.last_t) > 1e-9:
                self.states.append((t, px, py, yaw_deg, float(init), float(paused)))
                self.last_t = t
                updated = True

                # 限长
                MAX_N = 50000
                if len(self.states) > MAX_N:
                    self.states = self.states[-MAX_N:]

        if not updated:
            return

        # 更新 slider 范围
        max_idx = max(0, len(self.states) - 1)
        self.slider.setMaximum(max_idx)

        # LIVE 模式：始终显示最新，并把 slider 拉到末尾
        if self.live and not self.scrubbing:
            self.selected_idx = max_idx
            self.slider.blockSignals(True)
            self.slider.setValue(max_idx)
            self.slider.blockSignals(False)
            self.render_at_index(max_idx)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 600)
    w.show()
    sys.exit(app.exec_())
