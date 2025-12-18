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

def rotate_xy_about_origin(xs, ys, deg, x0, y0):
    """Rotate points (xs,ys) by deg around pivot (x0,y0). Returns (xr, yr)."""
    if xs is None or ys is None:
        return xs, ys
    if len(xs) == 0:
        return xs, ys
    th = math.radians(float(deg))
    c = math.cos(th); s = math.sin(th)
    dx = xs - x0
    dy = ys - y0
    xr = x0 + c*dx - s*dy
    yr = y0 + s*dx + c*dy
    return xr, yr

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Trajectory + Vehicle Heading Monitor")

        # UDP RX (telemetry)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", TELEMETRY_PORT))
        self.rx.setblocking(False)

        # UDP TX (control)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # === 多轨迹历史缓存：每个点 (t, x, y, yaw_deg, init, paused) ===
        self.states_eskf = []
        self.states_dr   = []
        self.states_gnss = []
        self.last_t = None

        # Scrub / live control
        self.live = True
        self.scrubbing = False
        self.selected_idx = -1

        # rate stats
        self.msg_count = 0
        self.last_rate_ts = time.time()

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "x")
        self.plot.setLabel("left", "y")

        # Legend + fixed colors
        self.legend = self.plot.addLegend()
        
        self.curve_eskf = self.plot.plot([], [], pen=pg.mkPen((50,150,255), width=2), name="ESKF")
        self.curve_dr   = self.plot.plot([], [], pen=pg.mkPen((255,120,50), width=2, style=QtCore.Qt.DashLine), name="DR")
        self.curve_gnss = self.plot.plot([], [], pen=pg.mkPen((80,220,120), width=2, style=QtCore.Qt.DotLine), name="GNSS")

        # Track render mode: line | points | both
        self.track_draw_mode = "line"

        self.draw_mode_combo = QtWidgets.QComboBox()
        self.draw_mode_combo.addItems(["Line", "Points", "Line + Points"])
        self.draw_mode_combo.currentIndexChanged.connect(self.on_draw_mode_changed)

        # Scatter items for each track (points mode)
        self.scat_eskf = pg.ScatterPlotItem(size=4, pen=None)
        self.scat_dr   = pg.ScatterPlotItem(size=4, pen=None)
        self.scat_gnss = pg.ScatterPlotItem(size=4, pen=None)
        self.plot.addItem(self.scat_eskf)
        self.plot.addItem(self.scat_dr)
        self.plot.addItem(self.scat_gnss)

        # 当前中心点
        self.pos_scatter = pg.ScatterPlotItem(size=8, pen=None)
        self.plot.addItem(self.pos_scatter)

        # 车体三角形
        self.vehicle_item = pg.PlotDataItem([], [], pen=None)
        self.plot.addItem(self.vehicle_item)

        # 车体形状（局部坐标）：车头朝 +X
        self.vehicle_scale = 0.1
        self.vehicle_shape_local = np.array([
            [ 1.0,  0.0],
            [-0.7,  0.5],
            [-0.7, -0.5],
            [ 1.0,  0.0],
        ], dtype=float) * self.vehicle_scale

        # View mode
        self.view_mode = "follow"   # follow | fit | fixed
        self.fixed_window = 40.0

        # ====== Measure tool state ======
        self.measure_mode = False
        self.measure_points = []   # [(x,y), (x,y)]
        self.measure_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(width=2))
        self.plot.addItem(self.measure_scatter)

        # Measure preview/final line
        self.measure_line = pg.PlotDataItem([], [], pen=pg.mkPen(width=1))
        self.plot.addItem(self.measure_line)

        # Keep last mouse pos in plot coordinates
        self.mouse_xy = None

        # Enable mouse tracking & clicks on the plot
        self.plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)

        # Right panel labels
        self.yaw_label = QtWidgets.QLabel("yaw: --- deg")
        self.yaw_label.setStyleSheet("font-size: 24px; font-weight: 600;")
        self.mode_label = QtWidgets.QLabel("mode: LIVE")
        self.state_label = QtWidgets.QLabel("state: ---")
        self.t_label = QtWidgets.QLabel("t: ---")
        self.pos_label = QtWidgets.QLabel("pos: ---")
        self.rate_label = QtWidgets.QLabel("rx: --- Hz")

        # Mouse / measure labels
        self.mouse_label = QtWidgets.QLabel("mouse: (---, ---)")
        self.measure_label = QtWidgets.QLabel("measure: ---")
        self.measure_label.setStyleSheet("font-size: 14px; font-weight: 600;")

        # Buttons
        btn_pause  = QtWidgets.QPushButton("Pause")
        btn_resume = QtWidgets.QPushButton("Resume")
        btn_quit   = QtWidgets.QPushButton("Quit")
        btn_clear  = QtWidgets.QPushButton("Clear Track")
        btn_back_now = QtWidgets.QPushButton("Back to Now")
        btn_measure = QtWidgets.QPushButton("Measure Distance")

        btn_pause.clicked.connect(lambda: self.send_cmd("pause"))
        btn_resume.clicked.connect(lambda: self.send_cmd("resume"))
        btn_quit.clicked.connect(lambda: self.send_cmd("quit"))
        btn_clear.clicked.connect(self.clear_track)
        btn_back_now.clicked.connect(self.back_to_now)
        btn_measure.clicked.connect(self.toggle_measure_mode)

        # Track visibility checkboxes
        self.cb_eskf = QtWidgets.QCheckBox("Show ESKF"); self.cb_eskf.setChecked(True)
        self.cb_dr   = QtWidgets.QCheckBox("Show DR");   self.cb_dr.setChecked(True)
        self.cb_gnss = QtWidgets.QCheckBox("Show GNSS"); self.cb_gnss.setChecked(True)
        self.cb_eskf.stateChanged.connect(self.redraw_current)
        self.cb_dr.stateChanged.connect(self.redraw_current)
        self.cb_gnss.stateChanged.connect(self.redraw_current)

        # Vehicle source selection
        self.pose_src_combo = QtWidgets.QComboBox()
        self.pose_src_combo.addItems([
            "Vehicle from ESKF",
            "Vehicle from DR",
            "Vehicle from GNSS (no heading)"
        ])
        self.pose_src_combo.currentIndexChanged.connect(self.redraw_current)

        # Per-track angle offsets (deg)
        self.off_eskf = QtWidgets.QDoubleSpinBox()
        self.off_dr   = QtWidgets.QDoubleSpinBox()
        self.off_gnss = QtWidgets.QDoubleSpinBox()
        for sp in (self.off_eskf, self.off_dr, self.off_gnss):
            sp.setRange(-180.0, 180.0)
            sp.setSingleStep(1.0)
            sp.setDecimals(1)
            sp.setValue(0.0)
            sp.valueChanged.connect(self.redraw_current)

        # View controls
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Auto Follow", "Fit All", "Fixed Window"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(5.0, 500.0)
        self.window_spin.setSingleStep(5.0)
        self.window_spin.setValue(self.fixed_window)
        self.window_spin.valueChanged.connect(self.on_window_changed)

        # Progress slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setTracking(True)
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider_info = QtWidgets.QLabel("0 / 0")

        # Layout
        right = QtWidgets.QVBoxLayout()

        right.addWidget(QtWidgets.QLabel("Track Draw Mode"))
        right.addWidget(self.draw_mode_combo)

        right.addWidget(self.yaw_label)
        right.addWidget(self.mode_label)
        right.addWidget(self.state_label)
        right.addWidget(self.t_label)
        right.addWidget(self.pos_label)
        right.addWidget(self.rate_label)

        right.addSpacing(6)
        right.addWidget(self.mouse_label)
        right.addWidget(self.measure_label)
        right.addWidget(btn_measure)

        right.addSpacing(10)
        right.addWidget(QtWidgets.QLabel("Progress (scrub history)"))
        right.addWidget(self.slider)
        right.addWidget(self.slider_info)
        right.addWidget(btn_back_now)

        right.addSpacing(10)
        right.addWidget(QtWidgets.QLabel("Tracks"))
        right.addWidget(self.cb_eskf)
        right.addWidget(self.cb_dr)
        right.addWidget(self.cb_gnss)
        right.addWidget(self.pose_src_combo)

        right.addSpacing(6)
        right.addWidget(QtWidgets.QLabel("Angle Offsets (deg)"))
        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(QtWidgets.QLabel("ESKF")); row1.addWidget(self.off_eskf)
        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(QtWidgets.QLabel("DR"));   row2.addWidget(self.off_dr)
        row3 = QtWidgets.QHBoxLayout(); row3.addWidget(QtWidgets.QLabel("GNSS")); row3.addWidget(self.off_gnss)
        right.addLayout(row1); right.addLayout(row2); right.addLayout(row3)

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

        # Timers
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(20)

        self.rate_timer = QtCore.QTimer(self)
        self.rate_timer.timeout.connect(self.update_rate)
        self.rate_timer.start(500)

    def on_draw_mode_changed(self, idx: int):
        if idx == 0:
            self.track_draw_mode = "line"
        elif idx == 1:
            self.track_draw_mode = "points"
        else:
            self.track_draw_mode = "both"
        self.redraw_current()


    # ===== Measure Tool =====
    def toggle_measure_mode(self):
        # 再次点击：重新开始新一轮测距（清空旧点与线）
        self.measure_mode = True
        self.measure_points = []
        self.measure_scatter.setData([], [])
        self.measure_line.setData([], [])
        self.measure_label.setText("measure: click point A")

    def on_mouse_moved(self, pos):
        vb = self.plot.getViewBox()
        if vb is None:
            return
        p = vb.mapSceneToView(pos)
        self.mouse_label.setText(f"mouse: ({p.x():.3f}, {p.y():.3f})")

    def on_mouse_clicked(self, evt):
        if not self.measure_mode:
            return
        if evt.button() != QtCore.Qt.LeftButton:
            return

        vb = self.plot.getViewBox()
        if vb is None:
            return
        p = vb.mapSceneToView(evt.scenePos())
        x, y = float(p.x()), float(p.y())

        self.measure_points.append((x, y))

        # 更新点显示
        xs = [pt[0] for pt in self.measure_points]
        ys = [pt[1] for pt in self.measure_points]
        self.measure_scatter.setData(xs, ys)

        if len(self.measure_points) == 1:
            self.measure_label.setText(f"measure: A=({x:.3f},{y:.3f})  click point B")
            # 先画一个零长度线，后续 on_mouse_moved 会更新成预览线
            self.measure_line.setData([x, x], [y, y])

        elif len(self.measure_points) == 2:
            ax, ay = self.measure_points[0]
            bx, by = self.measure_points[1]
            d = math.hypot(bx - ax, by - ay)
            self.measure_label.setText(
                f"measure: A=({ax:.3f},{ay:.3f})  B=({bx:.3f},{by:.3f})  dist={d:.3f}"
            )
            # 固化最终线
            self.measure_line.setData([ax, bx], [ay, by])
            # 选完两点自动退出（保留点与线）
            self.measure_mode = False

    # ===== Utils =====
    def send_cmd(self, cmd: str):
        msg = json.dumps({"cmd": cmd}, separators=(",", ":")).encode("utf-8")
        self.tx.sendto(msg, (CONTROL_IP, CONTROL_PORT))

    def on_mode_changed(self, idx: int):
        self.view_mode = "follow" if idx == 0 else ("fit" if idx == 1 else "fixed")
        self.redraw_current()

    def on_window_changed(self, v: float):
        self.fixed_window = float(v)
        self.redraw_current()

    def on_slider_pressed(self):
        self.scrubbing = True
        self.live = False
        self.mode_label.setText("mode: HISTORY")

    def on_slider_released(self):
        self.scrubbing = False

    def on_slider_value_changed(self, v: int):
        if self.max_len() <= 0:
            return
        self.selected_idx = int(np.clip(v, 0, self.max_len() - 1))
        if not self.live:
            self.render_at_index(self.selected_idx)

    def back_to_now(self):
        if self.max_len() <= 0:
            return
        self.live = True
        self.mode_label.setText("mode: LIVE")
        self.selected_idx = self.max_len() - 1
        self.slider.blockSignals(True)
        self.slider.setValue(self.selected_idx)
        self.slider.blockSignals(False)
        self.render_at_index(self.selected_idx)

    def clear_track(self):
        self.states_eskf.clear()
        self.states_dr.clear()
        self.states_gnss.clear()
        self.last_t = None

        self.curve_eskf.setData([], [])
        self.curve_dr.setData([], [])
        self.curve_gnss.setData([], [])
        self.pos_scatter.setData([])
        self.vehicle_item.setData([], [])

        # clear measure points too
        self.measure_points = []
        self.measure_scatter.setData([], [])
        self.measure_label.setText("measure: ---")
        self.measure_mode = False

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

    def max_len(self) -> int:
        return max(len(self.states_eskf), len(self.states_dr), len(self.states_gnss))

    def redraw_current(self):
        if self.max_len() <= 0:
            return
        idx = self.selected_idx
        if idx < 0:
            idx = self.max_len() - 1
        idx = int(np.clip(idx, 0, self.max_len() - 1))
        self.render_at_index(idx)

    def track_offset_deg(self, key: str) -> float:
        if key == "eskf": return float(self.off_eskf.value())
        if key == "dr":   return float(self.off_dr.value())
        if key == "gnss": return float(self.off_gnss.value())
        return 0.0

    # ===== Rendering =====
    def update_view(self, xs_list, ys_list, px: float, py: float):
        # Auto Follow: 只平移跟随，不改变缩放
        if self.view_mode == "follow":
            vb = self.plot.getViewBox()
            (x0, x1), (y0, y1) = vb.viewRange()
            w = x1 - x0
            h = y1 - y0
            if w < 1e-6 or h < 1e-6:
                half = self.fixed_window * 0.5
                vb.setRange(xRange=(px - half, px + half), yRange=(py - half, py + half), padding=0)
            else:
                vb.setRange(xRange=(px - w/2, px + w/2), yRange=(py - h/2, py + h/2), padding=0)
            return

        if self.view_mode == "fixed":
            half = self.fixed_window * 0.5
            self.plot.setXRange(px - half, px + half, padding=0)
            self.plot.setYRange(py - half, py + half, padding=0)
            return

        # fit: union
        xs_all = []
        ys_all = []
        for xs, ys in zip(xs_list, ys_list):
            if xs is None or ys is None or len(xs) < 2:
                continue
            xs_all.append(xs)
            ys_all.append(ys)
        if len(xs_all) == 0:
            return
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)

        xmin, xmax = float(np.min(xs_all)), float(np.max(xs_all))
        ymin, ymax = float(np.min(ys_all)), float(np.max(ys_all))
        padx = max(1.0, 0.1 * (xmax - xmin))
        pady = max(1.0, 0.1 * (ymax - ymin))
        self.plot.setXRange(xmin - padx, xmax + padx, padding=0)
        self.plot.setYRange(ymin - pady, ymax + pady, padding=0)

    def _curve_data_until(self, states, idx):
        if len(states) == 0:
            return None, None
        j = min(idx, len(states) - 1)
        arr = np.array(states[:j+1], dtype=float)
        return arr[:, 1], arr[:, 2]

    def _pose_at(self, states, idx):
        if len(states) == 0:
            return None
        j = min(idx, len(states) - 1)
        return states[j]

    def _track_pivot(self, key: str):
        st = self.states_eskf if key == "eskf" else (self.states_dr if key == "dr" else self.states_gnss)
        if len(st) == 0:
            return None
        return float(st[0][1]), float(st[0][2])

    def render_at_index(self, idx: int):
        idx = int(np.clip(idx, 0, self.max_len() - 1))

        xs_eskf = ys_eskf = None
        xs_dr   = ys_dr   = None
        xs_gnss = ys_gnss = None

        # -------- ESKF --------
        if self.cb_eskf.isChecked():
            xs_eskf, ys_eskf = self._curve_data_until(self.states_eskf, idx)
            if xs_eskf is not None and len(xs_eskf) > 0:
                x0, y0 = float(xs_eskf[0]), float(ys_eskf[0])
                xs_eskf, ys_eskf = rotate_xy_about_origin(
                    xs_eskf, ys_eskf, self.track_offset_deg("eskf"), x0, y0
                )
        # apply draw mode
        if (not self.cb_eskf.isChecked()) or xs_eskf is None or ys_eskf is None or len(xs_eskf) == 0:
            self.curve_eskf.setData([], [])
            self.scat_eskf.setData([], [])
        else:
            if self.track_draw_mode in ("line", "both"):
                self.curve_eskf.setData(xs_eskf, ys_eskf)
            else:
                self.curve_eskf.setData([], [])
            if self.track_draw_mode in ("points", "both"):
                self.scat_eskf.setData(xs_eskf, ys_eskf)
            else:
                self.scat_eskf.setData([], [])

        # -------- DR --------
        if self.cb_dr.isChecked():
            xs_dr, ys_dr = self._curve_data_until(self.states_dr, idx)
            if xs_dr is not None and len(xs_dr) > 0:
                x0, y0 = float(xs_dr[0]), float(ys_dr[0])
                xs_dr, ys_dr = rotate_xy_about_origin(
                    xs_dr, ys_dr, self.track_offset_deg("dr"), x0, y0
                )
        # apply draw mode
        if (not self.cb_dr.isChecked()) or xs_dr is None or ys_dr is None or len(xs_dr) == 0:
            self.curve_dr.setData([], [])
            self.scat_dr.setData([], [])
        else:
            if self.track_draw_mode in ("line", "both"):
                self.curve_dr.setData(xs_dr, ys_dr)
            else:
                self.curve_dr.setData([], [])
            if self.track_draw_mode in ("points", "both"):
                self.scat_dr.setData(xs_dr, ys_dr)
            else:
                self.scat_dr.setData([], [])

        # -------- GNSS --------
        if self.cb_gnss.isChecked():
            xs_gnss, ys_gnss = self._curve_data_until(self.states_gnss, idx)
            if xs_gnss is not None and len(xs_gnss) > 0:
                x0, y0 = float(xs_gnss[0]), float(ys_gnss[0])
                xs_gnss, ys_gnss = rotate_xy_about_origin(
                    xs_gnss, ys_gnss, self.track_offset_deg("gnss"), x0, y0
                )
        # apply draw mode
        if (not self.cb_gnss.isChecked()) or xs_gnss is None or ys_gnss is None or len(xs_gnss) == 0:
            self.curve_gnss.setData([], [])
            self.scat_gnss.setData([], [])
        else:
            if self.track_draw_mode in ("line", "both"):
                self.curve_gnss.setData(xs_gnss, ys_gnss)
            else:
                self.curve_gnss.setData([], [])
            if self.track_draw_mode in ("points", "both"):
                self.scat_gnss.setData(xs_gnss, ys_gnss)
            else:
                self.scat_gnss.setData([], [])

        # -------- choose pose source --------
        src = self.pose_src_combo.currentIndex()
        pose = None
        pose_is_gnss = False
        src_key = "eskf"

        if src == 0:
            src_key = "eskf"
            pose = self._pose_at(self.states_eskf, idx) or self._pose_at(self.states_dr, idx)
        elif src == 1:
            src_key = "dr"
            pose = self._pose_at(self.states_dr, idx) or self._pose_at(self.states_eskf, idx)
        else:
            src_key = "gnss"
            pose = self._pose_at(self.states_gnss, idx)
            pose_is_gnss = True
            if pose is None:
                pose = self._pose_at(self.states_eskf, idx) or self._pose_at(self.states_dr, idx)
                pose_is_gnss = False
                src_key = "eskf" if self._pose_at(self.states_eskf, idx) is not None else "dr"

        if pose is None:
            self.yaw_label.setText("yaw: --- deg")
            self.slider_info.setText(f"{idx+1} / {self.max_len()}")
            return

        t, px, py, yaw_deg, init, paused = pose

        # -------- apply offset to pose position & yaw (match rotated display) --------
        pivot = self._track_pivot(src_key)
        off = self.track_offset_deg(src_key)
        if pivot is not None:
            x0, y0 = pivot
            px_arr = np.array([px], dtype=float)
            py_arr = np.array([py], dtype=float)
            pxr, pyr = rotate_xy_about_origin(px_arr, py_arr, off, x0, y0)
            px, py = float(pxr[0]), float(pyr[0])

        yaw_deg = wrap_deg(yaw_deg + off)

        # -------- labels --------
        self.mode_label.setText("mode: LIVE" if self.live else "mode: HISTORY")
        self.slider_info.setText(f"{idx+1} / {self.max_len()}")

        self.t_label.setText(f"t: {t:.3f}")
        self.state_label.setText(f"state: {'PAUSED' if int(paused) else 'RUN'} | init: {int(init)}")
        self.pos_label.setText(f"pos: [{px:.3f}, {py:.3f}, 0.000]")

        if int(init) == 0 or pose_is_gnss:
            self.yaw_label.setText("yaw: --- deg")
        else:
            self.yaw_label.setText(f"yaw: {wrap_deg(yaw_deg):+.2f} deg")

        # center dot
        self.pos_scatter.setData([px], [py])

        # -------- vehicle triangle --------
        if pose_is_gnss or int(init) == 0:
            self.vehicle_item.setData([], [])
        else:
            theta = math.radians(yaw_deg)
            R = rot2d(theta)
            pts = (self.vehicle_shape_local @ R.T) + np.array([px, py], dtype=float)
            self.vehicle_item.setData(
                pts[:, 0], pts[:, 1],
                fillLevel=0,
                brush=pg.mkBrush(50, 150, 255, 120)
            )

        # -------- view update --------
        xs_list = [xs_eskf if self.cb_eskf.isChecked() else None,
                xs_dr   if self.cb_dr.isChecked()   else None,
                xs_gnss if self.cb_gnss.isChecked() else None]
        ys_list = [ys_eskf if self.cb_eskf.isChecked() else None,
                ys_dr   if self.cb_dr.isChecked()   else None,
                ys_gnss if self.cb_gnss.isChecked() else None]
        self.update_view(xs_list, ys_list, px, py)

    # ===== Telemetry parsing =====
    def _push_track(self, key: str, obj: dict, store: list, t: float, paused: int):
        tr = obj.get(key, None)
        if not isinstance(tr, dict):
            return

        init = int(tr.get("init", 0))
        x = float(tr.get("x", 0.0))
        y = float(tr.get("y", 0.0))
        yaw_rad = tr.get("yaw", 0.0)

        if init != 0 and yaw_rad is not None:
            yaw_deg = wrap_deg(float(yaw_rad) * 180.0 / math.pi)
        else:
            yaw_deg = 0.0

        store.append((t, x, y, yaw_deg, float(init), float(paused)))

        MAX_N = 50000
        if len(store) > MAX_N:
            del store[:len(store) - MAX_N]

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

            t = float(obj.get("t", 0.0))
            paused = int(obj.get("paused", 0))

            if self.last_t is not None and abs(t - self.last_t) <= 1e-9:
                continue

            self._push_track("eskf", obj, self.states_eskf, t, paused)
            self._push_track("dr",   obj, self.states_dr,   t, paused)
            self._push_track("gnss", obj, self.states_gnss, t, paused)

            self.last_t = t
            updated = True

        if not updated:
            return

        max_len = self.max_len()
        if max_len <= 0:
            return

        max_idx = max_len - 1
        self.slider.setMaximum(max_idx)

        if self.live and not self.scrubbing:
            self.selected_idx = max_idx
            self.slider.blockSignals(True)
            self.slider.setValue(max_idx)
            self.slider.blockSignals(False)
            self.render_at_index(max_idx)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1300, 720)
    w.show()
    sys.exit(app.exec_())
