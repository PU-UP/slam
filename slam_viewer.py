'''
CopyRight: 2020-2030, Positec Tech. CO.,LTD. All Rights Reserved.
FilePath: slam_viewer.py
Author: Zhengnan Pu/濮正楠 (Positec CN) zhengnan.pu@positecgroup.com
Date: 2025-04-21 18:54:33
Version: 0.1
LastEditTime: 2025-04-21 18:54:34
LastEditors: Zhengnan Pu/濮正楠 (Positec CN) zhengnan.pu@positecgroup.com
Description: 
'''
#!/usr/bin/env python3
import os
import argparse
import re
import bisect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.widgets import AxesWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# 日志->map提取

def parse_log_and_save_map(log_path, map_path):
    pattern = re.compile(r"index\s*:\s*(\d+),\s*t\s*:\s*([\-\d\.eE]+)\s+([\-\d\.eE]+)")
    with open(log_path, 'r', encoding='utf-8') as fin, \
         open(map_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            m = pattern.search(line)
            if m:
                _, x, y = m.groups()
                fout.write(f"0 {x} {y}\n")
    print(f"[+] Saved single map to {map_path}")

# 遍历子目录map提取

def find_map_paths(pg_dir):
    entries = []
    if not os.path.isdir(pg_dir):
        return entries
    for sub in sorted(os.listdir(pg_dir)):
        p = os.path.join(pg_dir, sub, 'path.txt')
        if os.path.isfile(p):
            entries.append((p, sub, False))
    return entries

# 解析map或path文件

def parse_map_file(path):
    xs, ys = [], []
    name = os.path.basename(path)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if name == 'path.txt' and len(parts) >= 2:
                x, y = parts[0], parts[1]
            elif len(parts) >= 3:
                _, x, y = parts[:3]
            else:
                continue
            xs.append(float(x)); ys.append(float(y))
    return xs, ys

# 提取回环信息

def extract_loops(log_path, output_path):
    conn_re = re.compile(r"curr_kf id: \{(\d+),(\d+)\}.*loop_kf id: \{(\d+),(\d+)\}")
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d+)")
    loops = []
    curr = None
    cnt = 0
    lines = open(log_path, 'r', encoding='utf-8').readlines()
    for i, line in enumerate(lines):
        m = conn_re.search(line)
        if m:
            curr = {'cur_traj': m.group(1), 'cur_id': m.group(2),
                    'loop_traj': m.group(3), 'loop_id': m.group(4)}
        if curr and 'Loop found' in line:
            update_time = None
            for j in range(i+1, len(lines)):
                if 'End of Optimization' in lines[j]:
                    mt = time_re.search(lines[j])
                    if mt: update_time = mt.group(1)
                    break
            if update_time:
                cnt += 1
                loops.append((cnt, curr['cur_traj'], curr['cur_id'],
                              curr['loop_traj'], curr['loop_id'], update_time))
            curr = None
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write('cnt cur_traj cur_id loop_traj loop_id update_time\n')
        for e in loops:
            fout.write(' '.join(map(str, e)) + '\n')
    print(f"[+] Extracted {len(loops)} loops to {output_path}")


def extract_wheel_slipping(log_path, output_path):
    """提取轮子打滑信息"""
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*?##### wheel slipping #####.*?anomaly_code\s*:\s*(\d+)\s*anomaly_code_from_viw_\s*:\s*(\d+)")
    
    with open(log_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        # 写入表头
        fout.write("time anomaly_code anomaly_code_from_viw_\n")
        
        # 读取并处理每一行
        for line in fin:
            match = pattern.search(line)
            if match:
                timestamp, anomaly_code, anomaly_code_from_viw = match.groups()
                # 只保留时间部分
                time_str = timestamp.split(' ')[1]
                fout.write(f"{time_str} {anomaly_code} {anomaly_code_from_viw}\n")
    
    print(f"[+] Extracted wheel slipping information to {output_path}")


def extract_map_operations(log_path, output_path):
    operations = []
    current_time = None
    current_type = None
    current_load_cnt = None
    current_update = None
    lines = []
    
    # 首先读取所有行到内存
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 步骤1: 查找MAP OPERATION标记
        if '---------- MAP OPEARTION -----------' in line:
            # 步骤2: 获取上一行的时间
            if i > 0:
                prev_line = lines[i-1]
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})', prev_line)
                if time_match:
                    current_time = time_match.group(1).split(' ')[1]  # 只保留时间部分
            
            # 步骤3: 查找操作类型
            i += 1
            while i < len(lines):
                if 'Map Operation Type:' in lines[i]:
                    type_match = re.search(r'Map Operation Type: (\w+)', lines[i])
                    if type_match:
                        current_type = type_match.group(1)
                    break
                i += 1
            
            # 步骤4: 查找MAP TO LOAD并获取加载数量
            while i < len(lines):
                if '**********MAP TO LOAD**********' in lines[i]:
                    i += 1
                    if i < len(lines):
                        if 'No map to load' in lines[i]:
                            current_load_cnt = 0
                        else:
                            load_match = re.search(r'(\d+) maps to load', lines[i])
                            if load_match:
                                current_load_cnt = int(load_match.group(1))
                    break
                i += 1
            
            # 步骤5: 查找MAP TO SAVE/UPDATE
            while i < len(lines):
                if '**********MAP TO SAVE/UPDATE**********' in lines[i]:
                    i += 1
                    if i < len(lines):
                        # 步骤6: 获取更新ID
                        update_match = re.search(r'Map id: \{(\d+)\}', lines[i])
                        if update_match:
                            current_update = update_match.group(1)
                            # 找到完整的操作信息，添加到列表
                            if current_time and current_type is not None and current_load_cnt is not None:
                                operations.append({
                                    'time': current_time,
                                    'type': current_type,
                                    'load_cnt': current_load_cnt,
                                    'update': current_update
                                })
                    break
                i += 1
            
            # 重置当前操作的状态
            current_time = None
            current_type = None
            current_load_cnt = None
            current_update = None
        i += 1
    
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for op in operations:
            # 格式化输出：时间 type load_cnt update
            f.write(f"{op['time']} {op['type']} {op['load_cnt']} {op['update']}\n")
    
    print(f"[+] Extracted {len(operations)} map operations to {output_path}")

# 提取pose信息
def parse_and_save_pose(log_path, pose_path):
    p = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}).*location/pose.*?([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    )
    with open(log_path, 'r', encoding='utf-8') as fin, \
         open(pose_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            m = p.search(line)
            if m:
                t, x, y, yaw, conf, cal, slip, reloc = m.groups()
                fout.write(f"{t} {x} {y} {yaw} {conf} {cal} {slip} {reloc}\n")
    print(f"[+] Saved poses to {pose_path}")
    
    # 在保存pose后提取MAP OPERATION信息
    extract_map_operations(log_path, 'mapping.log')

# 解析pose.txt

def parse_pose(pose_path):
    times, xs, ys, yaws = [], [], [], []
    confs, cals, slips, relocs = [], [], [], []
    for line in open(pose_path, 'r', encoding='utf-8'):
        parts = line.strip().split()
        if len(parts)==8:
            t, x, y, yaw, conf, cal, slip, reloc = parts
            times.append(t)
            xs.append(float(x))
            ys.append(float(y))
            yaws.append(float(yaw))
            confs.append(int(conf))
            cals.append(int(cal))
            slips.append(int(slip))
            relocs.append(int(reloc))
    times_s=[]
    for t in times:
        hh, mm, ss = t.split(':')
        times_s.append(int(hh)*3600 + int(mm)*60 + float(ss))
    return times, times_s, xs, ys, yaws, confs, cals, slips, relocs

class ComboBox(AxesWidget):
    def __init__(self, ax, label, options, value=None):
        super().__init__(ax)
        self.options = options
        # 统一时间格式，去掉毫秒
        if value:
            value = value.split('.')[0]
        self.value = value if value is not None else options[0]
        self.label = label
        self.callbacks = []
        
        # 设置样式
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(label)
        
        # 绘制当前值
        self.text = self.ax.text(0.5, 0.5, self.value,
                                ha='center', va='center',
                                transform=self.ax.transAxes)
        
        # 连接事件
        self.connect_event('button_press_event', self._on_click)
        
        # 创建选择窗口
        self.popup = None
        self.popup_ax = None
        
    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        # 如果已经有弹出窗口，先关闭
        if self.popup is not None:
            plt.close(self.popup)
            self.popup = None
            return
            
        # 创建弹出窗口
        self.popup = plt.figure(figsize=(4, 6))
        self.popup_ax = self.popup.add_subplot(111)
        
        # 绘制选项列表
        y_pos = np.arange(len(self.options))
        self.popup_ax.set_yticks(y_pos)
        self.popup_ax.set_yticklabels(self.options)
        self.popup_ax.set_xticks([])
        
        # 高亮当前值
        try:
            current_idx = self.options.index(self.value)
            self.popup_ax.axhspan(current_idx-0.4, current_idx+0.4,
                                 color='lightgray', alpha=0.5)
        except ValueError:
            pass  # 如果当前值不在选项中，不显示高亮
        
        # 连接点击事件
        self.popup.canvas.mpl_connect('button_press_event', self._on_select)
        
        plt.show()
        
    def _on_select(self, event):
        if event.inaxes != self.popup_ax:
            return
            
        idx = int(event.ydata + 0.5)
        if 0 <= idx < len(self.options):
            new_value = self.options[idx]
            if new_value != self.value:
                self.value = new_value
                self.text.set_text(self.value)
                self.ax.figure.canvas.draw_idle()
                self._notify_observers()
                
        plt.close(self.popup)
        self.popup = None
        
    def on_changed(self, func):
        """连接回调函数"""
        self.callbacks.append(func)
        
    def _notify_observers(self):
        """通知所有观察者"""
        for func in self.callbacks:
            func(self.value)
            
    def set_val(self, value):
        """设置新值"""
        if value:
            # 统一时间格式，去掉毫秒
            value = value.split('.')[0]
        if value in self.options:
            self.value = value
            self.text.set_text(value)
            self.ax.figure.canvas.draw_idle()

class TimeSelectionWindow:
    def __init__(self, parent, time_options, mapping_log_path, start_time, end_time):
        self.window = tk.Toplevel(parent)
        self.window.title("Time Selection")
        self.window.geometry("800x600")
        
        # 存储时间选项和映射信息
        self.time_options = time_options
        self.mapping_info = self._read_mapping_log(mapping_log_path)
        self.start_time = start_time
        self.end_time = end_time
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建文本显示区域
        self.text_frame = ttk.Frame(self.main_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_widget = tk.Text(self.text_frame, wrap=tk.WORD)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        # 显示mapping信息
        self._display_mapping_info()
        
        # 创建时间轴
        self.timeline_frame = ttk.Frame(self.main_frame)
        self.timeline_frame.pack(fill=tk.X, pady=10)
        
        # 创建时间轴滑块
        self.timeline = ttk.Scale(self.timeline_frame, from_=0, to=len(time_options)-1,
                                orient=tk.HORIZONTAL, length=600)
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 创建时间显示标签
        self.time_label = ttk.Label(self.timeline_frame, text="")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # 创建开始和结束时间选择
        self.time_selection_frame = ttk.Frame(self.main_frame)
        self.time_selection_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.time_selection_frame, text="Start Time:").pack(side=tk.LEFT)
        self.start_time_var = tk.StringVar(value=start_time)
        self.start_time_entry = ttk.Entry(self.time_selection_frame, textvariable=self.start_time_var)
        self.start_time_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.time_selection_frame, text="End Time:").pack(side=tk.LEFT)
        self.end_time_var = tk.StringVar(value=end_time)
        self.end_time_entry = ttk.Entry(self.time_selection_frame, textvariable=self.end_time_var)
        self.end_time_entry.pack(side=tk.LEFT, padx=5)
        
        # 创建按钮
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=10)
        
        self.confirm_button = ttk.Button(self.button_frame, text="Confirm", command=self._on_confirm)
        self.confirm_button.pack(side=tk.RIGHT, padx=5)
        
        self.cancel_button = ttk.Button(self.button_frame, text="Cancel", command=self._on_cancel)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # 绑定事件
        self.timeline.bind("<Motion>", self._on_timeline_move)
        self.timeline.bind("<ButtonRelease-1>", self._on_timeline_release)
        
        # 存储回调函数
        self.callback = None
        
        # 设置窗口关闭事件
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # 设置初始滑块位置
        if start_time in time_options:
            self.timeline.set(time_options.index(start_time))
        
    def _on_cancel(self):
        """处理取消按钮点击事件"""
        self.window.destroy()
        if self.callback:
            self.callback(None, None)  # 传递None表示取消操作
    
    def set_callback(self, callback):
        """设置回调函数"""
        self.callback = callback
    
    def _on_timeline_move(self, event):
        """处理时间轴移动事件"""
        idx = int(self.timeline.get())
        if 0 <= idx < len(self.time_options):
            current_time = self.time_options[idx]
            self.time_label.config(text=current_time)
            # 根据当前焦点更新输入框
            if self.start_time_entry.focus_get() == self.start_time_entry:
                self.start_time_var.set(current_time)
            elif self.end_time_entry.focus_get() == self.end_time_entry:
                self.end_time_var.set(current_time)
    
    def _on_timeline_release(self, event):
        """处理时间轴释放事件"""
        self._on_timeline_move(event)
    
    def _on_confirm(self):
        """处理确认按钮点击事件"""
        start_time = self.start_time_var.get()
        end_time = self.end_time_var.get()
        
        if start_time <= end_time:
            if self.callback:
                self.callback(start_time, end_time)
            self.window.destroy()
        else:
            tk.messagebox.showerror("Error", "Start time must be earlier than end time")

    def _read_mapping_log(self, mapping_log_path):
        """读取mapping.log文件内容"""
        mapping_info = []
        try:
            with open(mapping_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        time_str = parts[0].split('.')[0]  # 去掉毫秒
                        op_type = parts[1]
                        load_cnt = parts[2]
                        update_id = parts[3]
                        mapping_info.append({
                            'time': time_str,
                            'type': op_type,
                            'load_cnt': load_cnt,
                            'update': update_id
                        })
        except FileNotFoundError:
            print(f"Warning: {mapping_log_path} not found")
        return mapping_info
    
    def _display_mapping_info(self):
        """显示mapping信息"""
        self.text_widget.delete(1.0, tk.END)
        for info in self.mapping_info:
            line = f"Time: {info['time']} | Type: {info['type']} | Load Count: {info['load_cnt']} | Update ID: {info['update']}\n"
            self.text_widget.insert(tk.END, line)

class InteractiveViewer:
    SPEED_OPTIONS=[0.5,0.75,1,1.25,1.5,2,4,8]
    def __init__(self,map_entries,log_path,pose_path):
        self.map_entries=map_entries
        extract_loops(log_path,'loop_extracted.txt')
        parse_and_save_pose(log_path,pose_path)
        # 提取轮子打滑信息
        extract_wheel_slipping(log_path, 'slip.log')
        self.times,self.times_s,self.xs,self.ys,self.yaws,self.confs,self.cals,self.slips,self.relocs=parse_pose(pose_path)
        self.N=len(self.xs)
        self.playing=False; self.frame_float=0.0; self.frame=0
        self.speed_idx=self.SPEED_OPTIONS.index(1)
        self.tail_sec=5.0; self.full_history=False
        self.selected_loop_ids=[]
        self.loop_artists=[]; self.loops_visible=False
        
        # 轨迹显示状态
        self.traj_mode = 'none'  # none, conf, slip, reloc
        self.traj_lines = []  # 存储轨迹线

        # 读取mapping.log获取时间选项
        self.time_options = self._read_time_options('mapping.log')
        self.start_time = self.times[0]  # 默认开始时间
        self.end_time = self.times[-1]   # 默认结束时间
        
        # 日志显示状态
        self.showing_log = False
        self.log_text = None

        self.fig,self.ax=plt.subplots(figsize=(12,8))
        plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.95)
        
        # 添加键盘事件监听
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        cmap=plt.colormaps['tab10'](np.linspace(0, 1, len(map_entries)))
        for i,(mp,name,is_main) in enumerate(map_entries):
            mx,my=parse_map_file(mp)
            label=f"{name}{'*' if is_main else ''}"
            self.ax.plot(mx,my,'-',color=cmap[i],label=label)
        
        # 初始化轨迹线
        self.segment_line,=self.ax.plot([],[], '-',lw=2,color='lightgray',zorder=3)
        self.tail_line,=self.ax.plot([],[], '-',lw=3,color='orange',alpha=0.8,zorder=5)
        self.tri=plt.Polygon([[0,0],[0,0],[0,0]],closed=True,fc='red',ec='black',zorder=10)
        self.ax.add_patch(self.tri)
        
        # 轨迹信息显示
        ts0,ts1=self.times[0],self.times[-1]
        dur=self.times_s[-1]-self.times_s[0]
        h=int(dur//3600);m=int((dur%3600)//60);s=dur%60
        info=f"Data: {ts0}→{ts1}  Dur={h:02d}:{m:02d}:{s:06.3f}"
        self.info_text=self.ax.text(0.02,0.98,info,transform=self.ax.transAxes,
            ha='left',va='top',fontsize=11,
            bbox=dict(boxstyle='round',fc='w',ec='0.5',alpha=0.9))
            
        # 实时位置信息显示
        self.time_text=self.ax.text(0.02,0.90,'',transform=self.ax.transAxes,
            fontsize=11,bbox=dict(boxstyle='round',fc='w',ec='0.5',alpha=0.9))
        self.time_text.set_visible(False)
            
        self.ax.set_xlabel('X');self.ax.set_ylabel('Y')
        self.ax.grid(True);self.ax.set_aspect('equal','box');self.ax.legend(loc='upper right')

        # controls
        # 底部进度条
        ax_slider=self.fig.add_axes([0.1,0.02,0.85,0.03])
        self.slider=Slider(ax_slider,'Frame',0,self.N-1,valinit=0,valstep=1)
        self.slider.on_changed(self._on_slider_change)
        
        # 第一行控制按钮
        btn_y = 0.06
        btn_h = 0.04
        btn_w = 0.08
        btn_spacing = 0.06
        
        # 计算tail slider的最大值
        max_tail=min(40.0,self.times_s[-1]-self.times_s[0])
        
        ax_play=self.fig.add_axes([0.1,btn_y,btn_w,btn_h])
        self.btn_play=Button(ax_play,'Play')
        self.btn_play.on_clicked(self._toggle_play)
        
        ax_speed=self.fig.add_axes([0.1+btn_w+btn_spacing,btn_y,btn_w,btn_h])
        self.btn_speed=Button(ax_speed,f"Speed×{self.SPEED_OPTIONS[self.speed_idx]}")
        self.btn_speed.on_clicked(self._cycle_speed)
        
        ax_tail=self.fig.add_axes([0.1+2*(btn_w+btn_spacing),btn_y,0.25,btn_h])
        self.tail_slider=Slider(ax_tail,'Tail(s)',0.0,max_tail,valinit=self.tail_sec,valstep=0.5)
        self.tail_slider.on_changed(self._on_tail_change)
        
        ax_full=self.fig.add_axes([0.1+2*(btn_w+btn_spacing)+0.25+btn_spacing,btn_y,btn_w,btn_h])
        self.btn_full=Button(ax_full,'Full Hist')
        self.btn_full.on_clicked(self._toggle_full_history)
        
        # 顶部控制按钮
        btn_top_y = 0.96
        btn_top_w = 0.08
        btn_top_spacing = 0.06
        
        ax_show=self.fig.add_axes([0.1,btn_top_y,btn_top_w,btn_h])
        self.btn_show=Button(ax_show,'Show Traj')
        self.btn_show.on_clicked(self._toggle_show)
        
        ax_loopid=self.fig.add_axes([0.1+btn_top_w+btn_top_spacing,btn_top_y,btn_top_w,btn_h])
        self.box_loopid=TextBox(ax_loopid,'Loop ID','')
        self.box_loopid.on_submit(self._on_loopid_submit)
        
        ax_loop=self.fig.add_axes([0.1+2*(btn_top_w+btn_top_spacing),btn_top_y,btn_top_w,btn_h])
        self.btn_loop=Button(ax_loop,'Show Loops')
        self.btn_loop.on_clicked(self._toggle_loops)

        # 添加显示mapping.log按钮
        ax_log = self.fig.add_axes([0.1+3*(btn_top_w+btn_top_spacing), btn_top_y, btn_top_w, btn_h])
        self.btn_log = Button(ax_log, 'Show Log')
        self.btn_log.on_clicked(self._toggle_log)

        # 添加可编辑的时间输入框
        ax_start_time = self.fig.add_axes([0.1+4*(btn_top_w+btn_top_spacing), btn_top_y, btn_top_w, btn_h])
        self.start_time_box = TextBox(ax_start_time, 'Start', initial=self.start_time)
        self.start_time_box.on_submit(self._on_start_time_change)
        
        ax_end_time = self.fig.add_axes([0.1+5*(btn_top_w+btn_top_spacing), btn_top_y, btn_top_w, btn_h])
        self.end_time_box = TextBox(ax_end_time, 'End', initial=self.end_time)
        self.end_time_box.on_submit(self._on_end_time_change)
        
        ax_reset=self.fig.add_axes([0.1+6*(btn_top_w+btn_top_spacing),btn_top_y,btn_top_w,btn_h])
        self.btn_reset=Button(ax_reset,'Reset')
        self.btn_reset.on_clicked(self._reset_all)

        self._set_limits()
        self.ani=FuncAnimation(self.fig,self._update,interval=100,blit=False,repeat=True)

    def _read_time_options(self, mapping_log_path):
        """读取mapping.log中的时间选项"""
        time_options = set()
        try:
            with open(mapping_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    time_str = line.split()[0]  # 第一列是时间
                    # 统一时间格式，去掉毫秒
                    time_str = time_str.split('.')[0]
                    time_options.add(time_str)
        except FileNotFoundError:
            print(f"Warning: {mapping_log_path} not found")
        return sorted(list(time_options))

    def _on_start_time_change(self, text):
        """处理开始时间变化"""
        if text and text <= self.end_time:
            self.start_time = text
            self._update_display()
        else:
            self.start_time_box.set_val(self.start_time)

    def _on_end_time_change(self, text):
        """处理结束时间变化"""
        if text and text >= self.start_time:
            self.end_time = text
            self._update_display()
        else:
            self.end_time_box.set_val(self.end_time)

    def _update_display(self):
        """更新显示"""
        if self.traj_mode != 'none':
            self._update_trajectory()
        if self.loops_visible:
            self._toggle_loops(None)  # 重新显示回环
        self.fig.canvas.draw_idle()

    def _set_limits(self):
        all_x,all_y=[],[]
        for mp,_,_ in self.map_entries:
            xs,ys=parse_map_file(mp);all_x.extend(xs);all_y.extend(ys)
        all_x.extend(self.xs);all_y.extend(self.ys)
        if all_x and all_y:
            m=max(max(all_x)-min(all_x),max(all_y)-min(all_y))
            margin=m*0.1+0.5
            self.ax.set_xlim(min(all_x)-margin,max(all_x)+margin)
            self.ax.set_ylim(min(all_y)-margin,max(all_y)+margin)

    def _toggle_play(self,event):
        self.playing=not self.playing
        self.btn_play.label.set_text('Pause' if self.playing else 'Play')
        self._draw_frame()

    def _cycle_speed(self,event):
        self.speed_idx=(self.speed_idx+1)%len(self.SPEED_OPTIONS)
        self.btn_speed.label.set_text(f"Speed×{self.SPEED_OPTIONS[self.speed_idx]}")

    def _on_tail_change(self,val):
        self.tail_sec=val;self._draw_frame()

    def _toggle_full_history(self,event):
        self.full_history=not self.full_history
        self.btn_full.label.set_text('Tail Only' if self.full_history else 'Full Hist')
        self._draw_frame()

    def _on_slider_change(self,val):
        self.frame=int(val);self.frame_float=float(self.frame)
        self._draw_frame()

    def _toggle_show(self,event):
        # 切换轨迹显示模式
        if self.traj_mode == 'none':
            self.traj_mode = 'conf'
            self.btn_show.label.set_text('Show Slip')
        elif self.traj_mode == 'conf':
            self.traj_mode = 'slip'
            self.btn_show.label.set_text('Show Reloc')
        elif self.traj_mode == 'slip':
            self.traj_mode = 'reloc'
            self.btn_show.label.set_text('Hide Traj')
        else:
            self.traj_mode = 'none'
            self.btn_show.label.set_text('Show Traj')
        
        self._update_trajectory()

    def _toggle_loops(self,event):
        if self.loops_visible:
            for art in self.loop_artists: art.remove()
            self.loop_artists.clear()
            self.loops_visible=False
            self.btn_loop.label.set_text('Show Loops')
        else:
            # 获取时间范围内的索引
            start_idx, end_idx = self._get_time_range_indices()
            
            with open('loop_extracted.txt','r',encoding='utf-8') as f:
                next(f)
                for line in f:
                    cnt=int(line.split()[0])
                    if self.selected_loop_ids and cnt not in self.selected_loop_ids: continue
                    ut=line.split()[-1]
                    sec=self._time_to_seconds(ut)
                    idx=bisect.bisect_right(self.times_s,sec)
                    if start_idx <= idx < end_idx:  # 只显示时间范围内的回环
                        x,y=self.xs[idx],self.ys[idx]
                        art=self.ax.scatter(x,y,c='magenta',s=50,zorder=8)
                        txt=self.ax.text(x,y,str(cnt),color='magenta',fontsize=9)
                        self.loop_artists.extend([art,txt])
            self.loops_visible=True
            self.btn_loop.label.set_text('Hide Loops')
        self.fig.canvas.draw_idle()

    def _update_trajectory(self):
        # 清除现有的轨迹线
        for line in self.traj_lines:
            line.remove()
        self.traj_lines.clear()

        if self.traj_mode == 'none':
            self.segment_line.set_data([], [])
            self.fig.canvas.draw_idle()
            return

        # 获取时间范围内的索引
        start_idx, end_idx = self._get_time_range_indices()

        # 定义不同flag值的颜色
        colors = {
            'conf': {
                0: 'gray',      # 比lightgray深一点的灰色
                1: 'navy',      # 暗蓝色
                2: 'firebrick'  # 暗红色
            },
            'slip': {
                0: 'gray',      # 比lightgray深一点的灰色
                1: 'crimson'    # 深红色
            },
            'reloc': {
                0: 'gray',      # 比lightgray深一点的灰色
                1: 'gold',      # 金色
                2: 'orange',    # 橙色
                3: 'darkorange' # 深橙色
            }
        }

        # 获取当前flag类型的数据
        flag_data = {
            'conf': self.confs,
            'slip': self.slips,
            'reloc': self.relocs
        }[self.traj_mode]

        # 创建图例
        legend_elements = []
        current_colors = colors[self.traj_mode]

        # 为每个值创建图例
        for value, color in current_colors.items():
            if value in set(flag_data):  # 只显示实际存在的值
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, 
                                                label=f'{self.traj_mode}={value}'))

        # 绘制轨迹，根据flag值显示不同颜色的点
        segments = {}  # 按flag值分段的点
        for value in current_colors.keys():
            segments[value] = {'x': [], 'y': []}

        # 收集每个flag值对应的点（在时间范围内）
        for i in range(start_idx, end_idx):
            value = flag_data[i]
            if value in current_colors:
                segments[value]['x'].append(self.xs[i])
                segments[value]['y'].append(self.ys[i])

        # 绘制每个flag值的点
        for value, points in segments.items():
            if points['x']:  # 如果该值有对应的点
                line, = self.ax.plot(points['x'], points['y'], 'o', 
                                   color=current_colors[value],
                                   markersize=3,
                                   alpha=0.8)
                self.traj_lines.append(line)

        # 添加图例
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right')

        self.fig.canvas.draw_idle()

    def _reset_all(self, event):
        self.playing=False;self.btn_play.label.set_text('Play')
        self.frame=0;self.frame_float=0.0;self.slider.set_val(0)
        self.speed_idx=self.SPEED_OPTIONS.index(1)
        self.btn_speed.label.set_text(f"Speed×{self.SPEED_OPTIONS[self.speed_idx]}")
        self.tail_sec=5.0;self.tail_slider.set_val(self.tail_sec)
        self.full_history=False;self.btn_full.label.set_text('Full Hist')
        if self.loops_visible:
            for art in self.loop_artists: art.remove()
            self.loop_artists.clear();self.loops_visible=False;self.btn_loop.label.set_text('Show Loops')
        self.selected_loop_ids.clear();self.box_loopid.set_val('')
        
        # 重置轨迹显示
        self.traj_mode = 'none'
        self.btn_show.label.set_text('Show Traj')
        for line in self.traj_lines:
            line.remove()
        self.traj_lines.clear()
        self.segment_line.set_data([], [])
        
        # 重置时间选择
        self.start_time = self.times[0]
        self.end_time = self.times[-1]
        self.start_time_box.set_val(self.start_time)
        self.end_time_box.set_val(self.end_time)
        
        # 重置时显示Data信息，隐藏current信息
        self.info_text.set_visible(True)
        self.time_text.set_visible(False)
        
        # 如果正在显示日志，切换回图形显示
        if self.showing_log:
            self._toggle_log(None)
        
        # 更新显示
        self._update_display()
        self.fig.canvas.draw_idle()

    def _update(self,_):
        if self.playing:
            m=self.SPEED_OPTIONS[self.speed_idx]
            self.frame_float=(self.frame_float+m)%self.N
            self.frame=int(self.frame_float)
            prev=self.slider.eventson;self.slider.eventson=False
            self.slider.set_val(self.frame);self.slider.eventson=prev
            self._draw_frame()
        return []

    def _draw_frame(self):
        i=self.frame;x,y,yaw=self.xs[i],self.ys[i],self.yaws[i]
        head,width=0.2,0.06;pts=np.array([[0,width],[head,0],[0,-width]])
        R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        self.tri.set_xy(pts@R.T+np.array([x,y]))
        if self.full_history: idx0=0
        else: idx0=bisect.bisect_left(self.times_s,self.times_s[i]-self.tail_sec)
        self.tail_line.set_data(self.xs[idx0:i+1],self.ys[idx0:i+1])
        # 更新位置信息显示
        pose_info = f"Current: t={self.times[i]}\n[x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}]\n[conf={self.confs[i]}, cal={self.cals[i]}, slip={self.slips[i]}, reloc={self.relocs[i]}]"
        self.time_text.set_text(pose_info)
        self.time_text.set_visible(True)  # 显示current信息
        self.info_text.set_visible(False)  # 隐藏Data信息
        self.fig.canvas.draw_idle()

    def _time_to_seconds(self,t):
        hh,mm,ss=t.split(':');return int(hh)*3600+int(mm)*60+float(ss)

    def _get_time_range_indices(self):
        """获取选定时间范围内的索引"""
        start_idx = bisect.bisect_left(self.times, self.start_time)
        end_idx = bisect.bisect_right(self.times, self.end_time)
        return start_idx, end_idx

    def _parse_loop_ids(self,text):
        ids=set()
        for part in text.split(','):
            part=part.strip()
            if '-' in part:
                a,b=part.split('-')
                ids.update(range(int(a),int(b)+1))
            elif part.isdigit():
                ids.add(int(part))
        return ids

    def _on_loopid_submit(self,txt):
        self.selected_loop_ids=self._parse_loop_ids(txt)

    def _toggle_log(self, event):
        """切换显示mapping.log"""
        if not self.showing_log:
            # 保存当前图形状态
            self.ax.set_visible(False)
            
            # 创建文本框
            self.log_text = self.fig.text(0.5, 0.5, '', transform=self.fig.transFigure,
                                        ha='left', va='top', fontsize=10,
                                        bbox=dict(boxstyle='round', fc='w', ec='0.5', alpha=0.9))
            
            # 读取并显示mapping.log内容
            log_content = self._read_mapping_log('mapping.log')
            self.log_text.set_text(log_content)
            
            # 设置文本框位置和大小
            self.log_text.set_position((0.1, 0.9))  # 调整位置到左上角
            self.log_text.set_wrap(True)  # 启用自动换行
            self.log_text.set_clip_on(False)  # 允许文本超出图形边界
            
            self.showing_log = True
            self.btn_log.label.set_text('Hide Log')
        else:
            # 恢复图形显示
            self.ax.set_visible(True)
            if self.log_text:
                self.log_text.remove()
                self.log_text = None
            
            self.showing_log = False
            self.btn_log.label.set_text('Show Log')
            
            # 更新时间范围
            self._update_time_range()
        
        self.fig.canvas.draw_idle()

    def _read_mapping_log(self, mapping_log_path):
        """读取mapping.log和slip.log文件内容"""
        log_content = []
        has_mapping = False
        has_slip = False
        
        # 读取mapping.log
        try:
            with open(mapping_log_path, 'r', encoding='utf-8') as f:
                mapping_lines = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        time_str = parts[0].split('.')[0]  # 去掉毫秒
                        op_type = parts[1]
                        load_cnt = parts[2]
                        update_id = parts[3]
                        mapping_lines.append(f"{time_str} | {op_type} | Load: {load_cnt} | Update: {update_id}")
                
                if mapping_lines:
                    has_mapping = True
                    log_content.append("=== Mapping Operations ===")
                    log_content.extend(mapping_lines)
                else:
                    log_content.append("=== Mapping Operations ===")
                    log_content.append("No mapping information found.")
        except FileNotFoundError:
            log_content.append("=== Mapping Operations ===")
            log_content.append("No mapping information found.")
        
        # 读取slip.log
        try:
            with open('slip.log', 'r', encoding='utf-8') as f:
                slip_lines = []
                next(f)  # 跳过表头
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        time_str = parts[0]
                        anomaly_code = parts[1]
                        anomaly_code_from_viw = parts[2]
                        slip_lines.append(f"{time_str} | Anomaly: {anomaly_code} | From VIW: {anomaly_code_from_viw}")
                
                if slip_lines:
                    has_slip = True
                    if has_mapping:
                        log_content.append("")  # 添加空行分隔
                    log_content.append("=== Wheel Slipping Events ===")
                    log_content.extend(slip_lines)
                else:
                    if has_mapping:
                        log_content.append("")  # 添加空行分隔
                    log_content.append("=== Wheel Slipping Events ===")
                    log_content.append("No wheel slip detected.")
        except FileNotFoundError:
            if has_mapping:
                log_content.append("")  # 添加空行分隔
            log_content.append("=== Wheel Slipping Events ===")
            log_content.append("No wheel slip detected.")
        
        # 如果没有内容，显示提示信息
        if not log_content:
            return "No log information available."
        
        # 添加日志时间范围信息（使用pose.txt中的时间）
        if hasattr(self, 'times') and self.times:
            log_content.insert(0, f"Log Start: {self.times[0]}")
            log_content.append(f"Log End: {self.times[-1]}")
        
        return '\n'.join(log_content)

    def _update_time_range(self):
        """更新时间范围"""
        start_text = self.start_time_box.text
        end_text = self.end_time_box.text
        if start_text and end_text:
            self.start_time = start_text
            self.end_time = end_text
            self._update_display()

    def _on_key_press(self, event):
        """处理键盘事件"""
        if event.key == ' ':  # 空格键
            self._toggle_play(None)
        elif not self.playing:  # 只在暂停状态下响应方向键
            if event.key == 'left':  # 左方向键
                self.frame = max(0, self.frame - 1)
                self.frame_float = float(self.frame)
                self.slider.set_val(self.frame)
                self._draw_frame()
            elif event.key == 'right':  # 右方向键
                self.frame = min(self.N - 1, self.frame + 1)
                self.frame_float = float(self.frame)
                self.slider.set_val(self.frame)
                self._draw_frame()

    def run(self): plt.show()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['single','multi'],default='multi')
    parser.add_argument('--log','-l',default='log/slam_ordinary.log')
    parser.add_argument('--pose','-p',default='pose.txt')
    parser.add_argument('--pg-dir',default='map/pose_graph')
    args=parser.parse_args()
    if args.mode=='single':
        parse_log_and_save_map(args.log,'map.txt');maps=[('map.txt','map',True)]
    else: maps=find_map_paths(args.pg_dir)
    InteractiveViewer(maps,args.log,args.pose).run()


