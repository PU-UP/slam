# slam

```
# 构建工程
mkdir build

cd build

cmake .. && make -j

# 打开可视化界面
cd analysis_tools
python gui.py

# 运行程序
cd build
./app

# 更换参数
data文件夹中准备了4份数据，自带标定文件，可在config.yaml中切换
```



usage:
python ../analysis_tools/plot_xy.py eskf_result.txt gnss_result.txt --align_time 80