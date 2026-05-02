# config.py
# 机器人运行模式配置
# 支持切换真机/仿真模式

import os

# 机器人运行模式
ROBOT_MODE = os.getenv('ROBOT_MODE', 'real').lower()
# ROBOT_MODE = 'real'  # 真机模式
# ROBOT_MODE = 'simulation'  # 仿真模式

# 机械臂默认参数
PI_PORT = '/dev/ttyAMA0'
PI_BAUD = 115200

# 默认角度（归零位置）
DEFAULT_ANGLES = [0, 0, 0, 0, 0, 0]

# 默认坐标
DEFAULT_COORDS = [160, -100, 200, 0, 180, 90]

# 吸泵GPIO引脚
PUMP_PIN = 20
VALVE_PIN = 21

# 仿真模式配置
SIMULATION_CONFIG = {
    'enable_trajectory_logging': True,  # 是否记录轨迹
    'enable_visualization': True,        # 是否启用可视化
    'trajectory_log_file': 'temp/trajectory_log.json',  # 轨迹记录文件
    'playback_speed': 1.0,               # 回放速度
}

def is_simulation_mode():
    '''
    检查是否处于仿真模式
    '''
    return ROBOT_MODE == 'simulation'

def get_robot_mode():
    '''
    获取当前机器人运行模式
    '''
    return ROBOT_MODE

def set_robot_mode(mode):
    '''
    设置机器人运行模式
    mode: 'real' 或 'simulation'
    '''
    global ROBOT_MODE
    if mode.lower() in ['real', 'simulation']:
        ROBOT_MODE = mode.lower()
    else:
        raise ValueError(f"Invalid robot mode: {mode}. Must be 'real' or 'simulation'.")
