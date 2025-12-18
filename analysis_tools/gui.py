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
    # ---------- lifecycle ----------
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Trajectory + Vehicle Heading Monitor")

        self._init_network()
        self._init_state()
        self._init_plot()
        self._init_ui()
        self._init_timers()

    # ---------- init blocks ----------
    def _init_network(self):
        # UDP RX (telemetry)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", TELEMETRY_PORT))
        self.rx.setblocking(False)

        # UDP TX (control)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _init_state(self):
        # tracks: each point = (t, x, y, yaw_deg, init, paused)
        self.states = {
            "eskf": [],
            "dr":   [],
            "gnss": [],
        }
        self.last_t = None

        # scrub / live
        self.live = True
        self.scrubbing = False
        self.selected_idx = -1

        # rx rate stats
        self.msg_count = 0
        self.last_rate_ts = time.time()

        # draw mode: line | points | both
        self.track_draw_mode = "line"

        # view mode: follow | fit | fixed
        self.view_mode = "follow"
        self.fixed_window = 40.0  # used by follow init & fixed

        # vehicle shape
        self.vehicle_scale = 0.1
        self.vehicle_shape_local = np.array([
            [ 1.0,  0.0],
            [-0.7,  0.5],
            [-0.7, -0.5],
            [ 1.0,  0.0],
        ], dtype=float) * self.vehicle_scale

        # measure tool
        self.measure_mode = False
        self.measure_points = []  # [(x,y), (x,y)]

    def _init_plot(self):
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "x")
        self.plot.setLabel("left", "y")

        # legend
        self.legend = self.plot.addLegend()

        # unified per-track spec
        self.tracks = {
            "eskf": {
                "name": "ESKF",
                "color": (50, 150, 255),
                "style": None,
                "cb": None,       # QCheckBox (set in UI init)
                "off": None,      # QDoubleSpinBox
                "curve": None,    # PlotDataItem
                "scatter": None,  # ScatterPlotItem
            },
            "dr": {
                "name": "DR",
                "color": (255, 120, 50),
                "style": QtCore.Qt.DashLine,
                "cb": None,
                "off": None,
                "curve": None,
                "scatter": None,
            },
            "gnss": {
                "name": "GNSS",
                "color": (80, 220, 120),
                "style": QtCore.Qt.DotLine,
                "cb": None,
                "off": None,
                "curve": None,
                "scatter": None,
            }
        }

        # create plot items for tracks
        for key, spec in self.tracks.items():
            pen = pg.mkPen(spec["color"], width=2)
            if spec["style"] is not None:
                pen.setStyle(spec["style"])
            spec["curve"] = self.plot.plot([], [], pen=pen, name=spec["name"])
            spec["scatter"] = pg.ScatterPlotItem(size=4, pen=None)
            self.plot.addItem(spec["scatter"])

        # pose dot + vehicle
        self.pos_scatter = pg.ScatterPlotItem(size=8, pen=None)
        self.plot.addItem(self.pos_scatter)

        self.vehicle_item = pg.PlotDataItem([], [], pen=None)
        self.plot.addItem(self.vehicle_item)

        # measure visuals
        self.measure_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(width=2))
        self.plot.addItem(self.measure_scatter)

        self.measure_line = pg.PlotDataItem([], [], pen=pg.mkPen(width=1))
        self.plot.addItem(self.measure_line)

        # mouse interactions
        self.plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)

    def _init_ui(self):
        # --- labels ---
        self.yaw_label = QtWidgets.QLabel("yaw: --- deg")
        self.yaw_label.setStyleSheet("font-size: 24px; font-weight: 600;")

        self.mode_label  = QtWidgets.QLabel("mode: LIVE")
        self.state_label = QtWidgets.QLabel("state: ---")
        self.t_label     = QtWidgets.QLabel("t: ---")
        self.pos_label   = QtWidgets.QLabel("pos: ---")
        self.rate_label  = QtWidgets.QLabel("rx: --- Hz")

        self.mouse_label = QtWidgets.QLabel("mouse: (---, ---)")
        self.measure_label = QtWidgets.QLabel("measure: ---")
        self.measure_label.setStyleSheet("font-size: 14px; font-weight: 600;")

        # --- controls ---
        btn_pause   = QtWidgets.QPushButton("Pause")
        btn_resume  = QtWidgets.QPushButton("Resume")
        btn_quit    = QtWidgets.QPushButton("Quit")
        btn_clear   = QtWidgets.QPushButton("Clear Track")
        btn_measure = QtWidgets.QPushButton("Measure Distance")

        btn_pause.clicked.connect(lambda: self.send_cmd("pause"))
        btn_resume.clicked.connect(lambda: self.send_cmd("resume"))
        btn_quit.clicked.connect(lambda: self.send_cmd("quit"))
        btn_clear.clicked.connect(self.clear_track)
        btn_measure.clicked.connect(self.toggle_measure_mode)

        # draw mode
        self.draw_mode_combo = QtWidgets.QComboBox()
        self.draw_mode_combo.addItems(["Line", "Points", "Line + Points"])
        self.draw_mode_combo.currentIndexChanged.connect(self.on_draw_mode_changed)

        # view mode
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Auto Follow", "Fit All", "Fixed Window"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        # slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setTracking(True)
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider_info = QtWidgets.QLabel("0 / 0")

        # track checkboxes
        self.tracks["eskf"]["cb"] = QtWidgets.QCheckBox("Show ESKF"); self.tracks["eskf"]["cb"].setChecked(True)
        self.tracks["dr"]["cb"]   = QtWidgets.QCheckBox("Show DR");   self.tracks["dr"]["cb"].setChecked(True)
        self.tracks["gnss"]["cb"] = QtWidgets.QCheckBox("Show GNSS"); self.tracks["gnss"]["cb"].setChecked(True)

        for key in self.tracks:
            self.tracks[key]["cb"].stateChanged.connect(self.redraw_current)

        # vehicle source
        self.pose_src_combo = QtWidgets.QComboBox()
        self.pose_src_combo.addItems([
            "Vehicle from ESKF",
            "Vehicle from DR",
            "Vehicle from GNSS (no heading)"
        ])
        self.pose_src_combo.currentIndexChanged.connect(self.redraw_current)

        # per-track offsets
        self.tracks["eskf"]["off"] = QtWidgets.QDoubleSpinBox()
        self.tracks["dr"]["off"]   = QtWidgets.QDoubleSpinBox()
        self.tracks["gnss"]["off"] = QtWidgets.QDoubleSpinBox()
        for key in self.tracks:
            sp = self.tracks[key]["off"]
            sp.setRange(-180.0, 180.0)
            sp.setSingleStep(1.0)
            sp.setDecimals(1)
            sp.setValue(0.0)
            sp.valueChanged.connect(self.redraw_current)

        # --- layout ---
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

        right.addSpacing(10)
        right.addWidget(QtWidgets.QLabel("Tracks"))
        right.addWidget(self.tracks["eskf"]["cb"])
        right.addWidget(self.tracks["dr"]["cb"])
        right.addWidget(self.tracks["gnss"]["cb"])
        right.addWidget(self.pose_src_combo)

        right.addSpacing(6)
        right.addWidget(QtWidgets.QLabel("Angle Offsets (deg)"))
        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(QtWidgets.QLabel("ESKF")); row1.addWidget(self.tracks["eskf"]["off"])
        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(QtWidgets.QLabel("DR"));   row2.addWidget(self.tracks["dr"]["off"])
        row3 = QtWidgets.QHBoxLayout(); row3.addWidget(QtWidgets.QLabel("GNSS")); row3.addWidget(self.tracks["gnss"]["off"])
        right.addLayout(row1); right.addLayout(row2); right.addLayout(row3)

        right.addSpacing(10)
        right.addWidget(QtWidgets.QLabel("View Mode"))
        right.addWidget(self.mode_combo)

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

    def _init_timers(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(20)

        self.rate_timer = QtCore.QTimer(self)
        self.rate_timer.timeout.connect(self.update_rate)
        self.rate_timer.start(500)

    # ---------- UI handlers ----------
    def on_draw_mode_changed(self, idx: int):
        if idx == 0:
            self.track_draw_mode = "line"
        elif idx == 1:
            self.track_draw_mode = "points"
        else:
            self.track_draw_mode = "both"
        self.redraw_current()

    def on_mode_changed(self, idx: int):
        self.view_mode = "follow" if idx == 0 else ("fit" if idx == 1 else "fixed")
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

    # ---------- measure tool ----------
    def toggle_measure_mode(self):
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
        mx, my = float(p.x()), float(p.y())
        self.mouse_label.setText(f"mouse: ({mx:.3f}, {my:.3f})")

        # preview line after A point
        if self.measure_mode and len(self.measure_points) == 1:
            ax, ay = self.measure_points[0]
            self.measure_line.setData([ax, mx], [ay, my])

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

        xs = [pt[0] for pt in self.measure_points]
        ys = [pt[1] for pt in self.measure_points]
        self.measure_scatter.setData(xs, ys)

        if len(self.measure_points) == 1:
            self.measure_label.setText(f"measure: A=({x:.3f},{y:.3f})  click point B")
            self.measure_line.setData([x, x], [y, y])
        elif len(self.measure_points) == 2:
            ax, ay = self.measure_points[0]
            bx, by = self.measure_points[1]
            d = math.hypot(bx - ax, by - ay)
            self.measure_label.setText(
                f"measure: A=({ax:.3f},{ay:.3f})  B=({bx:.3f},{by:.3f})  dist={d:.3f}"
            )
            self.measure_line.setData([ax, bx], [ay, by])
            self.measure_mode = False  # keep points/line

    # ---------- control / misc ----------
    def send_cmd(self, cmd: str):
        msg = json.dumps({"cmd": cmd}, separators=(",", ":")).encode("utf-8")
        self.tx.sendto(msg, (CONTROL_IP, CONTROL_PORT))

    def update_rate(self):
        now = time.time()
        dt = now - self.last_rate_ts
        hz = self.msg_count / dt if dt > 1e-6 else 0.0
        self.rate_label.setText(f"rx: {hz:.1f} Hz")
        self.msg_count = 0
        self.last_rate_ts = now

    def clear_track(self):
        for key in self.states:
            self.states[key].clear()
        self.last_t = None

        for key, spec in self.tracks.items():
            spec["curve"].setData([], [])
            spec["scatter"].setData([], [])

        self.pos_scatter.setData([])
        self.vehicle_item.setData([], [])

        self.measure_mode = False
        self.measure_points = []
        self.measure_scatter.setData([], [])
        self.measure_line.setData([], [])
        self.measure_label.setText("measure: ---")

        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider_info.setText("0 / 0")

        self.mode_label.setText("mode: LIVE")
        self.live = True
        self.selected_idx = -1

    # ---------- data helpers ----------
    def max_len(self) -> int:
        return max(len(self.states["eskf"]), len(self.states["dr"]), len(self.states["gnss"]))

    def redraw_current(self):
        if self.max_len() <= 0:
            return
        idx = self.selected_idx
        if idx < 0:
            idx = self.max_len() - 1
        idx = int(np.clip(idx, 0, self.max_len() - 1))
        self.render_at_index(idx)

    def track_offset_deg(self, key: str) -> float:
        return float(self.tracks[key]["off"].value())

    def track_enabled(self, key: str) -> bool:
        return bool(self.tracks[key]["cb"].isChecked())

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
        st = self.states[key]
        if len(st) == 0:
            return None
        return float(st[0][1]), float(st[0][2])

    def _apply_draw_mode(self, key: str, xs, ys):
        spec = self.tracks[key]
        enabled = self.track_enabled(key)

        if (not enabled) or xs is None or ys is None or len(xs) == 0:
            spec["curve"].setData([], [])
            spec["scatter"].setData([], [])
            return

        if self.track_draw_mode in ("line", "both"):
            spec["curve"].setData(xs, ys)
        else:
            spec["curve"].setData([], [])

        if self.track_draw_mode in ("points", "both"):
            spec["scatter"].setData(xs, ys)
        else:
            spec["scatter"].setData([], [])

    # ---------- view ----------
    def update_view(self, xs_list, ys_list, px: float, py: float):
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

        # fit
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

    # ---------- render ----------
    def _render_tracks(self, idx: int):
        """Return rotated xs/ys dict for view-fit union."""
        xs_out = {"eskf": None, "dr": None, "gnss": None}
        ys_out = {"eskf": None, "dr": None, "gnss": None}

        for key in ("eskf", "dr", "gnss"):
            xs = ys = None
            if self.track_enabled(key):
                xs, ys = self._curve_data_until(self.states[key], idx)
                if xs is not None and len(xs) > 0:
                    x0, y0 = float(xs[0]), float(ys[0])
                    xs, ys = rotate_xy_about_origin(xs, ys, self.track_offset_deg(key), x0, y0)

            self._apply_draw_mode(key, xs, ys)
            xs_out[key], ys_out[key] = xs, ys

        return xs_out, ys_out

    def _select_pose(self, idx: int):
        """Pick pose based on pose_src_combo, return (pose, pose_is_gnss, src_key)."""
        src = self.pose_src_combo.currentIndex()

        if src == 0:  # ESKF
            pose = self._pose_at(self.states["eskf"], idx) or self._pose_at(self.states["dr"], idx)
            return pose, False, ("eskf" if self._pose_at(self.states["eskf"], idx) is not None else "dr")
        if src == 1:  # DR
            pose = self._pose_at(self.states["dr"], idx) or self._pose_at(self.states["eskf"], idx)
            return pose, False, ("dr" if self._pose_at(self.states["dr"], idx) is not None else "eskf")

        # GNSS (no heading)
        pose = self._pose_at(self.states["gnss"], idx)
        if pose is not None:
            return pose, True, "gnss"

        # fallback if no GNSS
        pose = self._pose_at(self.states["eskf"], idx) or self._pose_at(self.states["dr"], idx)
        if pose is None:
            return None, False, "eskf"
        return pose, False, ("eskf" if self._pose_at(self.states["eskf"], idx) is not None else "dr")

    def _apply_pose_offset(self, src_key: str, px, py, yaw_deg):
        """Rotate pose around its track pivot by offset, and add yaw offset."""
        pivot = self._track_pivot(src_key)
        off = self.track_offset_deg(src_key)

        if pivot is not None:
            x0, y0 = pivot
            px_arr = np.array([px], dtype=float)
            py_arr = np.array([py], dtype=float)
            pxr, pyr = rotate_xy_about_origin(px_arr, py_arr, off, x0, y0)
            px, py = float(pxr[0]), float(pyr[0])

        yaw_deg = wrap_deg(yaw_deg + off)
        return px, py, yaw_deg

    def render_at_index(self, idx: int):
        idx = int(np.clip(idx, 0, self.max_len() - 1))

        # tracks
        xs_map, ys_map = self._render_tracks(idx)

        # pose
        pose, pose_is_gnss, src_key = self._select_pose(idx)
        if pose is None:
            self.yaw_label.setText("yaw: --- deg")
            self.slider_info.setText(f"{idx+1} / {self.max_len()}")
            return

        t, px, py, yaw_deg, init, paused = pose
        px, py, yaw_deg = self._apply_pose_offset(src_key, px, py, yaw_deg)

        # labels
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

        # vehicle
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

        # view
        xs_list = [xs_map["eskf"] if self.track_enabled("eskf") else None,
                   xs_map["dr"]   if self.track_enabled("dr")   else None,
                   xs_map["gnss"] if self.track_enabled("gnss") else None]
        ys_list = [ys_map["eskf"] if self.track_enabled("eskf") else None,
                   ys_map["dr"]   if self.track_enabled("dr")   else None,
                   ys_map["gnss"] if self.track_enabled("gnss") else None]
        self.update_view(xs_list, ys_list, px, py)

    # ---------- telemetry ----------
    def _push_track(self, key: str, obj: dict, t: float, paused: int):
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

        self.states[key].append((t, x, y, yaw_deg, float(init), float(paused)))

        MAX_N = 50000
        if len(self.states[key]) > MAX_N:
            del self.states[key][:len(self.states[key]) - MAX_N]

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

            self._push_track("eskf", obj, t, paused)
            self._push_track("dr",   obj, t, paused)
            self._push_track("gnss", obj, t, paused)

            self.last_t = t
            updated = True

        if not updated:
            return

        max_len = self.max_len()
        if max_len <= 0:
            return

        max_idx = max_len - 1
        self.slider.setMaximum(max_idx)

        # LIVE: always latest (and keep slider at end)
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
