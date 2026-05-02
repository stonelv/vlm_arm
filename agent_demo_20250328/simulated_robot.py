# simulated_robot.py
# 模拟机器人接口，与真实MyCobot接口兼容
# 用于仿真模式和离线回放

import time
import json
from datetime import datetime
from config import SIMULATION_CONFIG, DEFAULT_ANGLES, DEFAULT_COORDS


class SimulatedRobot:
    '''
    模拟MyCobot机械臂接口，与真实接口保持兼容
    支持：
    1. 记录所有动作指令
    2. 回放记录的轨迹
    3. 可视化轨迹（可选）
    '''
    
    def __init__(self, port=None, baud=None, enable_logging=True, enable_visualization=None):
        '''
        初始化模拟机器人
        :param port: 串口端口（兼容真实接口，实际不使用）
        :param baud: 波特率（兼容真实接口，实际不使用）
        :param enable_logging: 是否启用轨迹记录
        :param enable_visualization: 是否启用可视化
        '''
        # 当前状态
        self.current_angles = DEFAULT_ANGLES.copy()
        self.current_coords = DEFAULT_COORDS.copy()
        self.fresh_mode = 0  # 默认插补模式
        
        # 轨迹记录
        self.enable_logging = enable_logging
        self.trajectory = []
        
        # 可视化
        if enable_visualization is None:
            enable_visualization = SIMULATION_CONFIG.get('enable_visualization', False)
        self.enable_visualization = enable_visualization
        
        print(f'[仿真模式] 模拟机器人已初始化')
        print(f'[仿真模式] 当前角度: {self.current_angles}')
        print(f'[仿真模式] 当前坐标: {self.current_coords}')
    
    def _log_action(self, action_type, **kwargs):
        '''
        记录动作到轨迹中
        :param action_type: 动作类型
        :param kwargs: 动作参数
        '''
        if not self.enable_logging:
            return
        
        action = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'params': kwargs,
            'state_before': {
                'angles': self.current_angles.copy(),
                'coords': self.current_coords.copy()
            }
        }
        
        self.trajectory.append(action)
        
        # 可视化输出
        if self.enable_visualization:
            self._visualize_action(action)
    
    def _visualize_action(self, action):
        '''
        可视化输出动作信息
        '''
        action_type = action['action_type']
        params = action['params']
        
        print(f'\n[仿真可视化] ------------------------')
        print(f'[仿真可视化] 动作类型: {action_type}')
        
        if action_type == 'send_angles':
            print(f'[仿真可视化] 目标角度: {params.get("angles")}')
            print(f'[仿真可视化] 速度: {params.get("speed")}')
        elif action_type == 'send_angle':
            print(f'[仿真可视化] 关节 {params.get("joint_index")}: {params.get("angle")} 度')
            print(f'[仿真可视化] 速度: {params.get("speed")}')
        elif action_type == 'send_coords':
            print(f'[仿真可视化] 目标坐标: {params.get("coords")}')
            print(f'[仿真可视化] 速度: {params.get("speed")}')
            print(f'[仿真可视化] 模式: {params.get("mode")}')
        elif action_type == 'release_all_servos':
            print(f'[仿真可视化] 释放所有伺服电机')
        elif action_type == 'set_fresh_mode':
            print(f'[仿真可视化] 设置运动模式: {params.get("mode")}')
        
        print(f'[仿真可视化] 当前状态:')
        print(f'[仿真可视化]   角度: {self.current_angles}')
        print(f'[仿真可视化]   坐标: {self.current_coords}')
        print(f'[仿真可视化] ------------------------\n')
    
    def send_angles(self, angles, speed):
        '''
        发送角度指令（与真实接口兼容）
        :param angles: 6个关节的角度列表
        :param speed: 运动速度
        '''
        self._log_action('send_angles', angles=angles, speed=speed)
        
        # 模拟运动过程
        print(f'[仿真模式] send_angles: angles={angles}, speed={speed}')
        
        # 更新当前状态
        self.current_angles = angles.copy()
        
        # 模拟运动时间（简化处理）
        time.sleep(0.1 * SIMULATION_CONFIG.get('playback_speed', 1.0))
    
    def send_angle(self, joint_index, angle, speed):
        '''
        发送单个关节角度指令（与真实接口兼容）
        :param joint_index: 关节索引（1-6）
        :param angle: 目标角度
        :param speed: 运动速度
        '''
        self._log_action('send_angle', joint_index=joint_index, angle=angle, speed=speed)
        
        # 模拟运动过程
        print(f'[仿真模式] send_angle: joint={joint_index}, angle={angle}, speed={speed}')
        
        # 更新当前状态
        if 1 <= joint_index <= 6:
            self.current_angles[joint_index - 1] = angle
        
        # 模拟运动时间
        time.sleep(0.1 * SIMULATION_CONFIG.get('playback_speed', 1.0))
    
    def send_coords(self, coords, speed, mode):
        '''
        发送坐标指令（与真实接口兼容）
        :param coords: 6维坐标 [x, y, z, rx, ry, rz]
        :param speed: 运动速度
        :param mode: 运动模式
        '''
        self._log_action('send_coords', coords=coords, speed=speed, mode=mode)
        
        # 模拟运动过程
        print(f'[仿真模式] send_coords: coords={coords}, speed={speed}, mode={mode}')
        
        # 更新当前状态
        self.current_coords = coords.copy()
        
        # 模拟运动时间
        time.sleep(0.1 * SIMULATION_CONFIG.get('playback_speed', 1.0))
    
    def release_all_servos(self):
        '''
        释放所有伺服电机（与真实接口兼容）
        '''
        self._log_action('release_all_servos')
        
        print(f'[仿真模式] release_all_servos: 释放所有伺服电机')
        
        # 模拟操作
        time.sleep(0.1 * SIMULATION_CONFIG.get('playback_speed', 1.0))
    
    def set_fresh_mode(self, mode):
        '''
        设置运动模式（与真实接口兼容）
        :param mode: 0=插补, 1=实时
        '''
        self._log_action('set_fresh_mode', mode=mode)
        
        print(f'[仿真模式] set_fresh_mode: mode={mode}')
        self.fresh_mode = mode
    
    def get_angles(self):
        '''
        获取当前角度（与真实接口兼容）
        :return: 6个关节的角度列表
        '''
        return self.current_angles.copy()
    
    def get_coords(self):
        '''
        获取当前坐标（与真实接口兼容）
        :return: 6维坐标列表 [x, y, z, rx, ry, rz]
        '''
        return self.current_coords.copy()
    
    def save_trajectory(self, file_path=None):
        '''
        保存轨迹记录到文件
        :param file_path: 文件路径，默认为配置中的路径
        '''
        if not self.enable_logging:
            print('[仿真模式] 轨迹记录未启用，无法保存')
            return
        
        if file_path is None:
            file_path = SIMULATION_CONFIG.get('trajectory_log_file', 'temp/trajectory_log.json')
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'start_time': datetime.now().isoformat(),
                'total_actions': len(self.trajectory),
                'trajectory': self.trajectory
            }, f, ensure_ascii=False, indent=2)
        
        print(f'[仿真模式] 轨迹已保存到: {file_path}')
        print(f'[仿真模式] 共记录 {len(self.trajectory)} 个动作')
    
    def load_trajectory(self, file_path=None):
        '''
        从文件加载轨迹记录
        :param file_path: 文件路径，默认为配置中的路径
        :return: 轨迹数据
        '''
        if file_path is None:
            file_path = SIMULATION_CONFIG.get('trajectory_log_file', 'temp/trajectory_log.json')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f'[仿真模式] 从 {file_path} 加载轨迹')
        print(f'[仿真模式] 共 {data.get("total_actions", 0)} 个动作')
        
        return data.get('trajectory', [])
    
    def playback_trajectory(self, trajectory=None, speed_multiplier=1.0):
        '''
        回放轨迹
        :param trajectory: 轨迹数据，如果为None则从默认文件加载
        :param speed_multiplier: 速度倍率
        '''
        if trajectory is None:
            trajectory = self.load_trajectory()
        
        print(f'\n[仿真模式] 开始回放轨迹...')
        print(f'[仿真模式] 共 {len(trajectory)} 个动作\n')
        
        for i, action in enumerate(trajectory):
            action_type = action['action_type']
            params = action['params']
            
            print(f'[仿真回放] [{i+1}/{len(trajectory)}] {action_type}')
            
            # 执行对应的动作
            if action_type == 'send_angles':
                self.send_angles(params['angles'], params['speed'])
            elif action_type == 'send_angle':
                self.send_angle(params['joint_index'], params['angle'], params['speed'])
            elif action_type == 'send_coords':
                self.send_coords(params['coords'], params['speed'], params['mode'])
            elif action_type == 'release_all_servos':
                self.release_all_servos()
            elif action_type == 'set_fresh_mode':
                self.set_fresh_mode(params['mode'])
            
            # 速度控制
            time.sleep(0.05 * speed_multiplier)
        
        print(f'\n[仿真模式] 轨迹回放完成')
    
    def get_state_summary(self):
        '''
        获取当前状态摘要
        :return: 状态摘要字典
        '''
        return {
            'current_angles': self.current_angles.copy(),
            'current_coords': self.current_coords.copy(),
            'fresh_mode': self.fresh_mode,
            'trajectory_length': len(self.trajectory)
        }


# 模拟GPIO类，用于仿真模式下的吸泵控制
class SimulatedGPIO:
    '''
    模拟RPi.GPIO接口，与真实接口保持兼容
    '''
    
    BCM = 'BCM'
    OUT = 'OUT'
    IN = 'IN'
    
    def __init__(self):
        self.mode = None
        self.warnings_enabled = True
        self.pin_states = {}
        self.pin_modes = {}
    
    def setwarnings(self, state):
        '''
        设置警告开关
        '''
        self.warnings_enabled = state
        print(f'[仿真GPIO] setwarnings: {state}')
    
    def setmode(self, mode):
        '''
        设置引脚编号模式
        '''
        self.mode = mode
        print(f'[仿真GPIO] setmode: {mode}')
    
    def setup(self, pin, mode, initial=None):
        '''
        设置引脚模式
        '''
        self.pin_modes[pin] = mode
        if initial is not None:
            self.pin_states[pin] = initial
        print(f'[仿真GPIO] setup: pin={pin}, mode={mode}, initial={initial}')
    
    def output(self, pin, state):
        '''
        设置引脚输出
        '''
        self.pin_states[pin] = state
        action = "开启" if state == 0 else "关闭"
        print(f'[仿真GPIO] output: pin={pin}, state={state} ({action})')
    
    def input(self, pin):
        '''
        读取引脚输入
        '''
        return self.pin_states.get(pin, 0)
    
    def cleanup(self, pin=None):
        '''
        清理GPIO设置
        '''
        if pin is None:
            self.pin_states.clear()
            self.pin_modes.clear()
            print('[仿真GPIO] cleanup: 清理所有引脚')
        else:
            self.pin_states.pop(pin, None)
            self.pin_modes.pop(pin, None)
            print(f'[仿真GPIO] cleanup: pin={pin}')


# 创建全局模拟GPIO实例（供utils_pump.py使用）
GPIO = SimulatedGPIO()
