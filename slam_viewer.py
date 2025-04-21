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

# 提取pose信息

def parse_and_save_pose(log_path, pose_path):
    p = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}).*location/pose.*?([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)"
    )
    with open(log_path, 'r', encoding='utf-8') as fin, \
         open(pose_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            m = p.search(line)
            if m:
                t, x, y, yaw = m.groups()
                fout.write(f"{t} {x} {y} {yaw}\n")
    print(f"[+] Saved poses to {pose_path}")

# 解析pose.txt

def parse_pose(pose_path):
    times, xs, ys, yaws = [], [], [], []
    for line in open(pose_path, 'r', encoding='utf-8'):
        parts = line.strip().split()
        if len(parts)==4:
            t, x, y, yaw = parts
            times.append(t); xs.append(float(x)); ys.append(float(y)); yaws.append(float(yaw))
    times_s=[]
    for t in times:
        hh, mm, ss = t.split(':')
        times_s.append(int(hh)*3600 + int(mm)*60 + float(ss))
    return times, times_s, xs, ys, yaws

class InteractiveViewer:
    SPEED_OPTIONS=[0.5,0.75,1,1.25,1.5,2,4,8]
    def __init__(self,map_entries,log_path,pose_path):
        self.map_entries=map_entries
        extract_loops(log_path,'loop_extracted.txt')
        parse_and_save_pose(log_path,pose_path)
        self.times,self.times_s,self.xs,self.ys,self.yaws=parse_pose(pose_path)
        self.N=len(self.xs)
        self.playing=False; self.frame_float=0.0; self.frame=0
        self.speed_idx=self.SPEED_OPTIONS.index(1)
        self.tail_sec=5.0; self.full_history=False
        self.selected_loop_ids=[]
        self.loop_artists=[]; self.loops_visible=False
        self.constraint_artists=[]; self.constraints_visible=False

        self.fig,self.ax=plt.subplots(figsize=(12,8))
        plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.95)
        cmap=plt.colormaps['tab10'](np.linspace(0, 1, len(map_entries)))
        for i,(mp,name,is_main) in enumerate(map_entries):
            mx,my=parse_map_file(mp)
            label=f"{name}{'*' if is_main else ''}"
            self.ax.plot(mx,my,'-',color=cmap[i],label=label)
        self.segment_line,=self.ax.plot([],[], '-',lw=2,color='green',zorder=3)
        self.tail_line,=self.ax.plot([],[], '-',lw=3,color='orange',alpha=0.8,zorder=5)
        self.tri=plt.Polygon([[0,0],[0,0],[0,0]],closed=True,fc='red',ec='black',zorder=10)
        self.ax.add_patch(self.tri)
        
        # 轨迹信息显示
        ts0,ts1=self.times[0],self.times[-1]
        dur=self.times_s[-1]-self.times_s[0]
        h=int(dur//3600);m=int((dur%3600)//60);s=dur%60
        info=f"Data: {ts0}→{ts1}  Dur={h:02d}:{m:02d}:{s:06.3f}"
        self.info_text=self.ax.text(0.02,0.96,info,transform=self.ax.transAxes,
            ha='left',va='center',fontsize=11,
            bbox=dict(boxstyle='round',fc='w',ec='0.5',alpha=0.9))
            
        # 实时位置信息显示
        self.time_text=self.ax.text(0.02,0.92,'',transform=self.ax.transAxes,
            fontsize=11,bbox=dict(boxstyle='round',fc='w',ec='0.5',alpha=0.9))
            
        self.ax.set_xlabel('X');self.ax.set_ylabel('Y')
        self.ax.grid(True);self.ax.set_aspect('equal','box');self.ax.legend(loc='upper right')

        # controls - 重新布局控制按钮
        # 底部进度条
        ax_slider=self.fig.add_axes([0.1,0.02,0.85,0.03])
        self.slider=Slider(ax_slider,'Frame',0,self.N-1,valinit=0,valstep=1)
        self.slider.on_changed(self._on_slider_change)
        
        # 第一行控制按钮
        btn_y = 0.06
        btn_h = 0.04
        btn_w = 0.08  # 减小按钮宽度
        btn_spacing = 0.03  # 增加按钮间距
        
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
        btn_top_w = 0.08  # 减小顶部按钮宽度
        btn_top_spacing = 0.03  # 增加顶部按钮间距
        
        ax_show=self.fig.add_axes([0.1,btn_top_y,btn_top_w,btn_h])
        self.btn_show=Button(ax_show,'Show Traj')
        self.btn_show.on_clicked(self._toggle_show)
        
        ax_loopid=self.fig.add_axes([0.1+btn_top_w+btn_top_spacing,btn_top_y,btn_top_w,btn_h])
        self.box_loopid=TextBox(ax_loopid,'Loop ID','')
        self.box_loopid.on_submit(self._on_loopid_submit)
        
        ax_loop=self.fig.add_axes([0.1+2*(btn_top_w+btn_top_spacing),btn_top_y,btn_top_w,btn_h])
        self.btn_loop=Button(ax_loop,'Show Loops')
        self.btn_loop.on_clicked(self._toggle_loops)
        
        ax_constraint=self.fig.add_axes([0.1+3*(btn_top_w+btn_top_spacing),btn_top_y,btn_top_w,btn_h])
        self.btn_constraint=Button(ax_constraint,'Show Const')
        self.btn_constraint.on_clicked(self._toggle_constraints)
        
        ax_reset=self.fig.add_axes([0.1+4*(btn_top_w+btn_top_spacing),btn_top_y,btn_top_w,btn_h])
        self.btn_reset=Button(ax_reset,'Reset')
        self.btn_reset.on_clicked(self._reset_all)

        self._set_limits()
        self.ani=FuncAnimation(self.fig,self._update,interval=100,blit=False,repeat=True)

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
        if self.segment_line.get_xdata():
            self.segment_line.set_data([],[])
            self.btn_show.label.set_text('Show Traj')
        else:
            self.segment_line.set_data(self.xs, self.ys)
            self.btn_show.label.set_text('Hide Traj')
        self.fig.canvas.draw_idle()

    def _toggle_loops(self,event):
        if self.loops_visible:
            for art in self.loop_artists: art.remove()
            self.loop_artists.clear()
            self.loops_visible=False
            self.btn_loop.label.set_text('Show Loops')
        else:
            with open('loop_extracted.txt','r',encoding='utf-8') as f:
                next(f)
                for line in f:
                    cnt=int(line.split()[0])
                    if self.selected_loop_ids and cnt not in self.selected_loop_ids: continue
                    ut=line.split()[-1]
                    sec=self._time_to_seconds(ut)
                    idx=bisect.bisect_right(self.times_s,sec)
                    if idx<self.N:
                        x,y=self.xs[idx],self.ys[idx]
                        art=self.ax.scatter(x,y,c='magenta',s=50,zorder=8)
                        txt=self.ax.text(x,y,str(cnt),color='magenta',fontsize=9)
                        self.loop_artists.extend([art,txt])
            self.loops_visible=True
            self.btn_loop.label.set_text('Hide Loops')
        self.fig.canvas.draw_idle()

    def _toggle_constraints(self,event):
        if self.constraints_visible:
            for art in self.constraint_artists: art.remove()
            self.constraint_artists.clear()
            self.constraints_visible=False
            self.btn_constraint.label.set_text('Show Const')
        else:
            # 读取loop_extracted.txt
            with open('loop_extracted.txt','r',encoding='utf-8') as f:
                next(f)  # 跳过标题行
                for line in f:
                    cnt,cur_traj,cur_id,loop_traj,loop_id,_=line.split()
                    if self.selected_loop_ids and int(cnt) not in self.selected_loop_ids:
                        continue
                    
                    # 获取当前帧位置
                    cur_idx=int(cur_id)
                    if cur_idx < self.N:
                        x1,y1=self.xs[cur_idx],self.ys[cur_idx]
                        
                        # 获取回环帧位置
                        loop_path=None
                        for mp,name,_ in self.map_entries:
                            if f"pose_graph_{loop_traj}" in name:
                                loop_path=mp
                                break
                        
                        if loop_path:
                            loop_xs,loop_ys=parse_map_file(loop_path)
                            loop_idx=int(loop_id)
                            if loop_idx < len(loop_xs):
                                x2,y2=loop_xs[loop_idx],loop_ys[loop_idx]
                                # 添加约束线
                                line=self.ax.plot([x1,x2],[y1,y2],'r--',alpha=0.5,linewidth=1,zorder=7)[0]
                                # 在约束线中点添加编号
                                mid_x,mid_y=(x1+x2)/2,(y1+y2)/2
                                txt=self.ax.text(mid_x,mid_y,str(cnt),color='red',fontsize=9,
                                               ha='center',va='center',bbox=dict(fc='white',ec='none',alpha=0.7))
                                self.constraint_artists.extend([line,txt])
            
            self.constraints_visible=True
            self.btn_constraint.label.set_text('Hide Const')
        self.fig.canvas.draw_idle()

    def _reset_all(self,event):
        self.playing=False;self.btn_play.label.set_text('Play')
        self.frame=0;self.frame_float=0.0;self.slider.set_val(0)
        self.speed_idx=self.SPEED_OPTIONS.index(1)
        self.btn_speed.label.set_text(f"Speed×{self.SPEED_OPTIONS[self.speed_idx]}")
        self.tail_sec=5.0;self.tail_slider.set_val(self.tail_sec)
        self.full_history=False;self.btn_full.label.set_text('Full Hist')
        self.segment_line.set_data([],[]);self.btn_show.label.set_text('Show Traj')
        if self.loops_visible:
            for art in self.loop_artists: art.remove()
            self.loop_artists.clear();self.loops_visible=False;self.btn_loop.label.set_text('Show Loops')
        if self.constraints_visible:
            for art in self.constraint_artists: art.remove()
            self.constraint_artists.clear();self.constraints_visible=False;self.btn_constraint.label.set_text('Show Const')
        self.selected_loop_ids.clear();self.box_loopid.set_val('')
        self._draw_frame()

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
        pose_info = f"Current: t={self.times[i]}  [x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}]"
        self.time_text.set_text(pose_info)
        self.fig.canvas.draw_idle()

    def _time_to_seconds(self,t):
        hh,mm,ss=t.split(':');return int(hh)*3600+int(mm)*60+float(ss)

    def run(self): plt.show()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['single','multi'],default='multi')
    parser.add_argument('--log','-l',default='visual_log/log/slam_ordinary.log')
    parser.add_argument('--pose','-p',default='pose.txt')
    parser.add_argument('--pg-dir',default='visual_log/map/pose_graph')
    args=parser.parse_args()
    if args.mode=='single':
        parse_log_and_save_map(args.log,'map.txt');maps=[('map.txt','map',True)]
    else: maps=find_map_paths(args.pg_dir)
    InteractiveViewer(maps,args.log,args.pose).run()
