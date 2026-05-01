#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全动作执行器 - Safe Action Executor
========================================
功能：
1. 接收大模型生成的高层指令（抓取/放置/移动到位姿）
2. 转换为机械臂API调用
3. 安全约束：限位、速度/加速度约束、碰撞前停止、异常回滚
4. 统一接口：支持仿真模式与真机模式
5. 完整的日志记录

作者：AI Assistant
日期：2026-05-01
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('safe_action_executor.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger('SafeActionExecutor')


# ============================================================================
# 数据类型定义
# ============================================================================

class ActionType(Enum):
    """动作类型枚举"""
    MOVE_TO_POSE = "move_to_pose"      # 移动到位姿
    PICK = "pick"                        # 抓取
    PLACE = "place"                      # 放置
    MOVE_JOINTS = "move_joints"          # 移动关节
    HOME = "home"                        # 归零
    RELAX = "relax"                      # 放松


@dataclass
class Pose:
    """位姿数据类 - 笛卡尔坐标系"""
    x: float = 0.0      # X坐标 (mm)
    y: float = 0.0      # Y坐标 (mm)
    z: float = 0.0      # Z坐标 (mm)
    rx: float = 0.0     # 绕X轴旋转 (deg)
    ry: float = 180.0   # 绕Y轴旋转 (deg)
    rz: float = 90.0    # 绕Z轴旋转 (deg)
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]
    
    @classmethod
    def from_list(cls, coords: List[float]) -> 'Pose':
        return cls(
            x=coords[0], y=coords[1], z=coords[2],
            rx=coords[3], ry=coords[4], rz=coords[5]
        )
    
    def distance_to(self, other: 'Pose') -> float:
        """计算与另一个位姿的欧氏距离"""
        return np.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )
    
    def __repr__(self) -> str:
        return f"Pose(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f}, " \
               f"rx={self.rx:.1f}, ry={self.ry:.1f}, rz={self.rz:.1f})"


@dataclass
class JointAngles:
    """关节角度数据类"""
    joints: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def __getitem__(self, index: int) -> float:
        return self.joints[index]
    
    def __len__(self) -> int:
        return len(self.joints)
    
    def to_list(self) -> List[float]:
        return self.joints.copy()
    
    @classmethod
    def from_list(cls, angles: List[float]) -> 'JointAngles':
        return cls(joints=angles.copy())
    
    def max_diff(self, other: 'JointAngles') -> float:
        """计算与另一个关节角度的最大差值"""
        return max(abs(a - b) for a, b in zip(self.joints, other.joints))
    
    def __repr__(self) -> str:
        return f"JointAngles({[round(a, 2) for a in self.joints]})"


@dataclass
class HighLevelCommand:
    """高层指令数据类 - 来自大模型的指令"""
    action_type: ActionType
    target_pose: Optional[Pose] = None
    target_joints: Optional[JointAngles] = None
    height: Optional[float] = None           # 抓取/放置高度
    safe_height: float = 220.0               # 安全移动高度
    speed: int = 30                           # 速度百分比 (1-100)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        parts = [f"HighLevelCommand(action={self.action_type.value}"]
        if self.target_pose:
            parts.append(f", pose={self.target_pose}")
        if self.target_joints:
            parts.append(f", joints={self.target_joints}")
        if self.height:
            parts.append(f", height={self.height}")
        parts.append(f", speed={self.speed})")
        return "".join(parts)


# ============================================================================
# 安全约束定义
# ============================================================================

@dataclass
class SafetyConstraints:
    """安全约束配置"""
    
    # 关节限位 (单位: 度) - MyCobot 280 M5
    joint_limits: List[tuple] = field(default_factory=lambda: [
        (-165.0, 165.0),   # 关节1
        (-90.0, 90.0),     # 关节2
        (-165.0, 165.0),   # 关节3
        (-90.0, 90.0),     # 关节4
        (-165.0, 165.0),   # 关节5
        (-180.0, 180.0),   # 关节6
    ])
    
    # 工作空间限制 (单位: mm) - 笛卡尔坐标系
    workspace_limits: Dict[str, tuple] = field(default_factory=lambda: {
        'x': (-280.0, 280.0),
        'y': (-280.0, 280.0),
        'z': (0.0, 350.0),
    })
    
    # 速度限制
    max_speed: int = 80               # 最大速度百分比
    min_speed: int = 5                 # 最小速度百分比
    default_speed: int = 30            # 默认速度
    
    # 加速度限制 (简化为速度变化率)
    max_speed_change: float = 20.0     # 每次运动的最大速度变化
    
    # 碰撞检测阈值
    collision_distance_threshold: float = 50.0  # 碰撞预警距离 (mm)
    emergency_stop_distance: float = 20.0       # 紧急停止距离 (mm)
    
    # 超时设置
    move_timeout: float = 15.0         # 单次移动超时 (秒)
    total_timeout: float = 120.0       # 总任务超时 (秒)
    
    # 安全高度
    default_safe_height: float = 220.0  # 默认安全移动高度


# ============================================================================
# 异常定义
# ============================================================================

class SafetyException(Exception):
    """安全相关异常"""
    pass


class JointLimitExceeded(SafetyException):
    """关节限位超限"""
    pass


class WorkspaceLimitExceeded(SafetyException):
    """工作空间超限"""
    pass


class CollisionDetected(SafetyException):
    """检测到碰撞"""
    pass


class TimeoutException(SafetyException):
    """操作超时"""
    pass


class ActionFailedException(SafetyException):
    """动作执行失败"""
    pass


# ============================================================================
# 机械臂接口抽象类
# ============================================================================

class RobotArmInterface(ABC):
    """机械臂接口抽象类 - 定义统一接口"""
    
    def __init__(self, name: str = "RobotArm"):
        self.name = name
        self._connected = False
        self._current_pose = Pose()
        self._current_joints = JointAngles()
        self._is_moving = False
        self._gripper_activated = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def connect(self) -> bool:
        """连接机械臂"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def is_moving(self) -> bool:
        return self._is_moving
    
    @abstractmethod
    def get_current_pose(self) -> Pose:
        """获取当前位姿"""
        pass
    
    @abstractmethod
    def get_current_joints(self) -> JointAngles:
        """获取当前关节角度"""
        pass
    
    @abstractmethod
    def move_to_pose(self, pose: Pose, speed: int = 30, wait: bool = True) -> bool:
        """移动到指定位姿"""
        pass
    
    @abstractmethod
    def move_to_joints(self, joints: JointAngles, speed: int = 30, wait: bool = True) -> bool:
        """移动到指定关节角度"""
        pass
    
    @abstractmethod
    def activate_gripper(self, activate: bool) -> bool:
        """激活/释放吸泵/夹爪"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """立即停止运动"""
        pass
    
    @abstractmethod
    def wait_for_stop(self, timeout: float = 10.0) -> bool:
        """等待运动停止"""
        pass
    
    def home(self, speed: int = 40) -> bool:
        """归零位置"""
        home_joints = JointAngles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self.move_to_joints(home_joints, speed)
    
    def relax(self) -> bool:
        """放松所有关节"""
        self.logger.info("Relaxing all joints")
        self.stop()
        self._is_moving = False
        return True


# ============================================================================
# 仿真机械臂实现
# ============================================================================

class SimulatedRobotArm(RobotArmInterface):
    """仿真机械臂实现 - 用于测试和调试"""
    
    def __init__(self, name: str = "SimulatedArm"):
        super().__init__(name)
        self._move_duration = 2.0  # 仿真移动持续时间
        self.logger.info(f"Created {name} in SIMULATION mode")
    
    def connect(self) -> bool:
        self.logger.info("Connecting to simulated robot arm...")
        time.sleep(0.5)
        self._connected = True
        self._current_pose = Pose(x=150, y=0, z=220, rx=0, ry=180, rz=90)
        self._current_joints = JointAngles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.logger.info("Simulated robot arm connected successfully")
        return True
    
    def disconnect(self) -> bool:
        self.logger.info("Disconnecting simulated robot arm...")
        self._connected = False
        self.logger.info("Simulated robot arm disconnected")
        return True
    
    def get_current_pose(self) -> Pose:
        return self._current_pose
    
    def get_current_joints(self) -> JointAngles:
        return self._current_joints
    
    def move_to_pose(self, pose: Pose, speed: int = 30, wait: bool = True) -> bool:
        if not self._connected:
            self.logger.error("Not connected to robot")
            return False
        
        self.logger.info(f"[SIM] Moving to pose: {pose} at speed {speed}%")
        self._is_moving = True
        
        if wait:
            # 仿真移动过程
            time.sleep(self._move_duration)
            self._current_pose = Pose(
                x=pose.x, y=pose.y, z=pose.z,
                rx=pose.rx, ry=pose.ry, rz=pose.rz
            )
            self._is_moving = False
            self.logger.info(f"[SIM] Reached pose: {self._current_pose}")
        
        return True
    
    def move_to_joints(self, joints: JointAngles, speed: int = 30, wait: bool = True) -> bool:
        if not self._connected:
            self.logger.error("Not connected to robot")
            return False
        
        self.logger.info(f"[SIM] Moving to joints: {joints} at speed {speed}%")
        self._is_moving = True
        
        if wait:
            time.sleep(self._move_duration)
            self._current_joints = JointAngles.from_list(joints.to_list())
            self._is_moving = False
            self.logger.info(f"[SIM] Reached joints: {self._current_joints}")
        
        return True
    
    def activate_gripper(self, activate: bool) -> bool:
        if not self._connected:
            self.logger.error("Not connected to robot")
            return False
        
        action = "ACTIVATING" if activate else "DEACTIVATING"
        self.logger.info(f"[SIM] {action} gripper")
        time.sleep(0.3)
        self._gripper_activated = activate
        self.logger.info(f"[SIM] Gripper {'activated' if activate else 'deactivated'}")
        return True
    
    def stop(self) -> bool:
        self.logger.info("[SIM] Stopping movement immediately")
        self._is_moving = False
        return True
    
    def wait_for_stop(self, timeout: float = 10.0) -> bool:
        self.logger.info(f"[SIM] Waiting for stop (timeout: {timeout}s)")
        start_time = time.time()
        while self._is_moving and (time.time() - start_time < timeout):
            time.sleep(0.1)
        return not self._is_moving


# ============================================================================
# 真机机械臂实现 (MyCobot)
# ============================================================================

class MyCobotRobotArm(RobotArmInterface):
    """真机机械臂实现 - MyCobot 280 M5"""
    
    def __init__(self, port: str = None, baud: int = 115200, name: str = "MyCobot"):
        super().__init__(name)
        self._port = port
        self._baud = baud
        self._mc = None
        self._gpio_initialized = False
        self.logger.info(f"Created {name} in REAL mode")
    
    def connect(self) -> bool:
        try:
            self.logger.info("Connecting to MyCobot robot arm...")
            
            # 尝试导入 pymycobot
            try:
                from pymycobot.mycobot import MyCobot
                from pymycobot import PI_PORT, PI_BAUD
            except ImportError:
                self.logger.error("pymycobot not installed. Cannot connect to real robot.")
                self.logger.info("Falling back to simulation mode for testing.")
                return False
            
            # 使用默认端口如果未指定
            port = self._port or PI_PORT
            baud = self._baud or PI_BAUD
            
            self._mc = MyCobot(port, baud)
            self._mc.set_fresh_mode(0)  # 插补模式
            
            # 初始化GPIO (用于吸泵控制)
            self._init_gpio()
            
            self._connected = True
            
            # 读取当前状态
            self._current_pose = self.get_current_pose()
            self._current_joints = self.get_current_joints()
            
            self.logger.info(f"MyCobot connected successfully on {port}:{baud}")
            self.logger.info(f"Current pose: {self._current_pose}")
            self.logger.info(f"Current joints: {self._current_joints}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MyCobot: {e}")
            self._connected = False
            return False
    
    def _init_gpio(self):
        """初始化GPIO用于吸泵控制"""
        try:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(20, GPIO.OUT)
            GPIO.setup(21, GPIO.OUT)
            GPIO.output(20, 1)  # 默认关闭吸泵电磁阀
            self._gpio_initialized = True
            self.logger.info("GPIO initialized for gripper control")
        except Exception as e:
            self.logger.warning(f"GPIO initialization failed (not on Raspberry Pi?): {e}")
            self._gpio_initialized = False
    
    def disconnect(self) -> bool:
        self.logger.info("Disconnecting MyCobot...")
        self.stop()
        
        if self._gpio_initialized:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
                self.logger.info("GPIO cleaned up")
            except Exception as e:
                self.logger.warning(f"GPIO cleanup failed: {e}")
        
        self._connected = False
        self._mc = None
        self.logger.info("MyCobot disconnected")
        return True
    
    def get_current_pose(self) -> Pose:
        if not self._connected or not self._mc:
            return self._current_pose
        
        try:
            coords = self._mc.get_coords()
            if coords and len(coords) >= 6:
                self._current_pose = Pose.from_list(coords)
        except Exception as e:
            self.logger.warning(f"Failed to get coords: {e}")
        
        return self._current_pose
    
    def get_current_joints(self) -> JointAngles:
        if not self._connected or not self._mc:
            return self._current_joints
        
        try:
            angles = self._mc.get_angles()
            if angles and len(angles) >= 6:
                self._current_joints = JointAngles.from_list(angles)
        except Exception as e:
            self.logger.warning(f"Failed to get angles: {e}")
        
        return self._current_joints
    
    def move_to_pose(self, pose: Pose, speed: int = 30, wait: bool = True) -> bool:
        if not self._connected or not self._mc:
            self.logger.error("Not connected to robot")
            return False
        
        speed = max(1, min(100, speed))
        coords = pose.to_list()
        
        self.logger.info(f"Moving to pose: {pose} at speed {speed}%")
        self._is_moving = True
        
        try:
            # mode=0 表示插补模式
            self._mc.send_coords(coords, speed, 0)
            
            if wait:
                return self.wait_for_stop()
            return True
            
        except Exception as e:
            self.logger.error(f"Move to pose failed: {e}")
            self._is_moving = False
            return False
    
    def move_to_joints(self, joints: JointAngles, speed: int = 30, wait: bool = True) -> bool:
        if not self._connected or not self._mc:
            self.logger.error("Not connected to robot")
            return False
        
        speed = max(1, min(100, speed))
        angles = joints.to_list()
        
        self.logger.info(f"Moving to joints: {joints} at speed {speed}%")
        self._is_moving = True
        
        try:
            self._mc.send_angles(angles, speed)
            
            if wait:
                return self.wait_for_stop()
            return True
            
        except Exception as e:
            self.logger.error(f"Move to joints failed: {e}")
            self._is_moving = False
            return False
    
    def activate_gripper(self, activate: bool) -> bool:
        if not self._connected:
            self.logger.error("Not connected to robot")
            return False
        
        try:
            if activate:
                # 开启吸泵
                self.logger.info("Activating gripper (pump ON)")
                if self._gpio_initialized:
                    import RPi.GPIO as GPIO
                    GPIO.output(20, 0)
                time.sleep(0.3)
                self._gripper_activated = True
                self.logger.info("Gripper activated")
            else:
                # 关闭吸泵并泄气
                self.logger.info("Deactivating gripper (pump OFF + release)")
                if self._gpio_initialized:
                    import RPi.GPIO as GPIO
                    GPIO.output(20, 1)  # 关闭吸泵电磁阀
                    time.sleep(0.05)
                    GPIO.output(21, 0)  # 打开泄气阀门
                    time.sleep(0.2)
                    GPIO.output(21, 1)
                    time.sleep(0.05)
                    # 再一次泄气，确保物体释放
                    GPIO.output(21, 0)
                    time.sleep(0.2)
                    GPIO.output(21, 1)
                self._gripper_activated = False
                self.logger.info("Gripper deactivated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Gripper operation failed: {e}")
            return False
    
    def stop(self) -> bool:
        if self._mc:
            try:
                self.logger.info("Stopping robot immediately")
                self._mc.stop()
            except Exception as e:
                self.logger.warning(f"Stop command failed: {e}")
        
        self._is_moving = False
        return True
    
    def wait_for_stop(self, timeout: float = 10.0) -> bool:
        if not self._mc:
            self._is_moving = False
            return True
        
        start_time = time.time()
        check_interval = 0.1
        
        while time.time() - start_time < timeout:
            try:
                # 检查是否还在移动
                is_moving = self._mc.is_moving()
                if not is_moving:
                    self._is_moving = False
                    # 更新当前状态
                    self.get_current_pose()
                    self.get_current_joints()
                    return True
            except Exception as e:
                self.logger.warning(f"is_moving check failed: {e}")
            
            time.sleep(check_interval)
        
        self.logger.warning(f"Wait for stop timed out after {timeout}s")
        self._is_moving = True  # 假设仍在移动
        return False


# ============================================================================
# 安全检查器
# ============================================================================

class SafetyChecker:
    """安全检查器 - 负责所有安全约束的验证"""
    
    def __init__(self, constraints: Optional[SafetyConstraints] = None):
        self.constraints = constraints or SafetyConstraints()
        self.logger = logging.getLogger('SafetyChecker')
        # 简单的障碍物列表 - 实际应用中可扩展为更复杂的碰撞检测
        self._obstacles: List[Dict[str, Any]] = []
    
    def add_obstacle(self, pose: Pose, radius: float = 50.0, name: str = "obstacle"):
        """添加障碍物用于碰撞检测"""
        self._obstacles.append({
            'pose': pose,
            'radius': radius,
            'name': name
        })
        self.logger.info(f"Added obstacle: {name} at {pose}, radius={radius}mm")
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        self._obstacles.clear()
        self.logger.info("All obstacles cleared")
    
    def check_joint_limits(self, joints: JointAngles) -> bool:
        """检查关节角度是否在限位范围内"""
        for i, (angle, (min_limit, max_limit)) in enumerate(
            zip(joints.joints, self.constraints.joint_limits)
        ):
            if not (min_limit <= angle <= max_limit):
                raise JointLimitExceeded(
                    f"Joint {i+1} angle {angle:.1f}° exceeds limits [{min_limit:.1f}°, {max_limit:.1f}°]"
                )
        return True
    
    def check_workspace(self, pose: Pose) -> bool:
        """检查位姿是否在工作空间内"""
        limits = self.constraints.workspace_limits
        
        if not (limits['x'][0] <= pose.x <= limits['x'][1]):
            raise WorkspaceLimitExceeded(
                f"X coordinate {pose.x:.1f}mm exceeds limits [{limits['x'][0]:.1f}, {limits['x'][1]:.1f}]"
            )
        
        if not (limits['y'][0] <= pose.y <= limits['y'][1]):
            raise WorkspaceLimitExceeded(
                f"Y coordinate {pose.y:.1f}mm exceeds limits [{limits['y'][0]:.1f}, {limits['y'][1]:.1f}]"
            )
        
        if not (limits['z'][0] <= pose.z <= limits['z'][1]):
            raise WorkspaceLimitExceeded(
                f"Z coordinate {pose.z:.1f}mm exceeds limits [{limits['z'][0]:.1f}, {limits['z'][1]:.1f}]"
            )
        
        return True
    
    def check_collision(self, current_pose: Pose, target_pose: Pose) -> tuple:
        """
        检查从当前位姿到目标位姿的路径是否会碰撞
        返回: (is_safe, warning_distance, nearest_obstacle)
        """
        min_distance = float('inf')
        nearest_obstacle = None
        
        # 简单的直线碰撞检测 - 在起点和终点之间采样
        num_samples = 5
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pose = Pose(
                x=current_pose.x + t * (target_pose.x - current_pose.x),
                y=current_pose.y + t * (target_pose.y - current_pose.y),
                z=current_pose.z + t * (target_pose.z - current_pose.z),
                rx=target_pose.rx,
                ry=target_pose.ry,
                rz=target_pose.rz
            )
            
            for obstacle in self._obstacles:
                obs_pose = obstacle['pose']
                obs_radius = obstacle['radius']
                distance = sample_pose.distance_to(obs_pose)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_obstacle = obstacle
                
                # 紧急停止距离检查
                if distance < self.constraints.emergency_stop_distance:
                    raise CollisionDetected(
                        f"Imminent collision with obstacle '{obstacle['name']}' at distance {distance:.1f}mm"
                    )
        
        # 检查是否需要预警
        if min_distance < self.constraints.collision_distance_threshold:
            self.logger.warning(
                f"Warning: Approaching obstacle '{nearest_obstacle['name'] if nearest_obstacle else 'unknown'}' "
                f"at distance {min_distance:.1f}mm"
            )
            return False, min_distance, nearest_obstacle
        
        return True, min_distance, nearest_obstacle
    
    def check_speed(self, speed: int) -> int:
        """检查并约束速度在安全范围内"""
        constrained_speed = max(
            self.constraints.min_speed,
            min(speed, self.constraints.max_speed)
        )
        
        if constrained_speed != speed:
            self.logger.warning(
                f"Speed {speed}% constrained to {constrained_speed}% "
                f"(limits: [{self.constraints.min_speed}, {self.constraints.max_speed}])"
            )
        
        return constrained_speed
    
    def validate_move(self, current_pose: Pose, target_pose: Pose, 
                       current_joints: JointAngles, target_joints: JointAngles,
                       speed: int) -> Dict[str, Any]:
        """
        执行完整的移动验证
        返回验证结果字典，安全违规时抛出特定异常
        """
        result = {
            'safe': True,
            'warnings': [],
            'constrained_speed': speed,
            'collision_distance': float('inf'),
        }
        
        # 检查关节限位（失败时直接抛出异常）
        self.check_joint_limits(target_joints)
        
        # 检查工作空间（失败时直接抛出异常）
        self.check_workspace(target_pose)
        
        # 检查碰撞（失败时直接抛出异常）
        is_safe, distance, obstacle = self.check_collision(current_pose, target_pose)
        result['collision_distance'] = distance
        if not is_safe and distance < self.constraints.collision_distance_threshold:
            warning_msg = f"Approaching obstacle at {distance:.1f}mm"
            result['warnings'].append(warning_msg)
            self.logger.warning(warning_msg)
        
        # 约束速度
        result['constrained_speed'] = self.check_speed(speed)
        
        if result['warnings']:
            self.logger.warning(f"Move validation warnings: {result['warnings']}")
        
        return result


# ============================================================================
# 动作历史记录器 (用于回滚)
# ============================================================================

@dataclass
class ActionRecord:
    """动作记录 - 用于回滚"""
    action_type: ActionType
    start_pose: Pose
    start_joints: JointAngles
    end_pose: Optional[Pose] = None
    end_joints: Optional[JointAngles] = None
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    gripper_state: Optional[bool] = None  # None表示未改变
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActionHistory:
    """动作历史管理器"""
    
    def __init__(self, max_history: int = 50):
        self._history: List[ActionRecord] = []
        self._max_history = max_history
        self.logger = logging.getLogger('ActionHistory')
    
    def record_start(self, action_type: ActionType, 
                     current_pose: Pose, current_joints: JointAngles,
                     gripper_state: Optional[bool] = None,
                     **metadata) -> ActionRecord:
        """记录动作开始"""
        record = ActionRecord(
            action_type=action_type,
            start_pose=Pose(
                x=current_pose.x, y=current_pose.y, z=current_pose.z,
                rx=current_pose.rx, ry=current_pose.ry, rz=current_pose.rz
            ),
            start_joints=JointAngles.from_list(current_joints.to_list()),
            gripper_state=gripper_state,
            metadata=metadata
        )
        self._history.append(record)
        
        # 限制历史长度
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        self.logger.debug(f"Started action: {action_type.value}")
        return record
    
    def record_end(self, record: ActionRecord, success: bool,
                   end_pose: Optional[Pose] = None,
                   end_joints: Optional[JointAngles] = None):
        """记录动作结束"""
        record.success = success
        if end_pose:
            record.end_pose = Pose(
                x=end_pose.x, y=end_pose.y, z=end_pose.z,
                rx=end_pose.rx, ry=end_pose.ry, rz=end_pose.rz
            )
        if end_joints:
            record.end_joints = JointAngles.from_list(end_joints.to_list())
        
        status = "SUCCESS" if success else "FAILED"
        self.logger.debug(f"Ended action: {record.action_type.value} - {status}")
    
    def get_last_successful_action(self) -> Optional[ActionRecord]:
        """获取最近成功的动作"""
        for record in reversed(self._history):
            if record.success:
                return record
        return None
    
    def get_rollback_sequence(self) -> List[ActionRecord]:
        """获取回滚序列 (从最近到最早的成功动作)"""
        return [r for r in reversed(self._history) if r.success]
    
    def clear(self):
        """清空历史"""
        self._history.clear()
        self.logger.info("Action history cleared")
    
    @property
    def history(self) -> List[ActionRecord]:
        return self._history.copy()


# ============================================================================
# 指令解析器
# ============================================================================

class CommandParser:
    """高层指令解析器 - 将自然语言或结构化指令转换为可执行命令"""
    
    def __init__(self):
        self.logger = logging.getLogger('CommandParser')
    
    def parse_from_dict(self, data: Dict[str, Any]) -> HighLevelCommand:
        """从字典解析指令"""
        action_str = data.get('action', '').lower()
        
        # 解析动作类型
        action_type_map = {
            'move': ActionType.MOVE_TO_POSE,
            'move_to_pose': ActionType.MOVE_TO_POSE,
            'goto': ActionType.MOVE_TO_POSE,
            'pick': ActionType.PICK,
            'grasp': ActionType.PICK,
            'place': ActionType.PLACE,
            'put': ActionType.PLACE,
            'move_joints': ActionType.MOVE_JOINTS,
            'joints': ActionType.MOVE_JOINTS,
            'home': ActionType.HOME,
            'zero': ActionType.HOME,
            'relax': ActionType.RELAX,
            'release': ActionType.RELAX,
        }
        
        action_type = action_type_map.get(action_str, ActionType.MOVE_TO_POSE)
        
        # 解析目标位姿
        target_pose = None
        if 'pose' in data:
            pose_data = data['pose']
            if isinstance(pose_data, list) and len(pose_data) >= 3:
                target_pose = Pose(
                    x=float(pose_data[0]),
                    y=float(pose_data[1]),
                    z=float(pose_data[2]),
                    rx=float(pose_data[3]) if len(pose_data) > 3 else 0.0,
                    ry=float(pose_data[4]) if len(pose_data) > 4 else 180.0,
                    rz=float(pose_data[5]) if len(pose_data) > 5 else 90.0,
                )
            elif isinstance(pose_data, dict):
                target_pose = Pose(
                    x=float(pose_data.get('x', 0)),
                    y=float(pose_data.get('y', 0)),
                    z=float(pose_data.get('z', 0)),
                    rx=float(pose_data.get('rx', 0)),
                    ry=float(pose_data.get('ry', 180)),
                    rz=float(pose_data.get('rz', 90)),
                )
        
        # 解析目标关节
        target_joints = None
        if 'joints' in data:
            joints_data = data['joints']
            if isinstance(joints_data, list):
                target_joints = JointAngles.from_list([float(a) for a in joints_data])
        
        # 解析高度参数
        height = data.get('height')
        safe_height = data.get('safe_height', 220.0)
        speed = data.get('speed', 30)
        
        # 其他参数
        params = {k: v for k, v in data.items() 
                  if k not in ['action', 'pose', 'joints', 'height', 'safe_height', 'speed']}
        
        command = HighLevelCommand(
            action_type=action_type,
            target_pose=target_pose,
            target_joints=target_joints,
            height=height,
            safe_height=safe_height,
            speed=speed,
            params=params
        )
        
        self.logger.info(f"Parsed command: {command}")
        return command
    
    def parse_from_llm_response(self, llm_text: str) -> HighLevelCommand:
        """
        从大模型的自然语言响应解析指令
        这是一个简化版本，实际应用中可能需要更复杂的NLP或结构化输出
        """
        # 这里假设LLM返回的是结构化的JSON或可解析的格式
        # 实际应用中可能需要结合正则表达式或更复杂的解析
        
        import re
        import json
        
        # 尝试提取JSON
        json_match = re.search(r'\{[\s\S]*\}', llm_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self.parse_from_dict(data)
            except json.JSONDecodeError:
                pass
        
        # 简单的关键词解析作为后备
        data = {}
        
        # 动作类型
        llm_lower = llm_text.lower()
        if any(word in llm_lower for word in ['抓取', '吸取', 'pick', 'grasp']):
            data['action'] = 'pick'
        elif any(word in llm_lower for word in ['放置', '放下', 'place', 'put']):
            data['action'] = 'place'
        elif any(word in llm_lower for word in ['移动', '前往', 'move', 'goto']):
            data['action'] = 'move'
        elif any(word in llm_lower for word in ['归零', 'home', 'zero']):
            data['action'] = 'home'
        else:
            data['action'] = 'move'
        
        # 尝试提取坐标
        coords = re.findall(r'[xyzXYZ]?[:=]?\s*(-?\d+(?:\.\d+)?)', llm_text)
        if len(coords) >= 3:
            data['pose'] = [float(c) for c in coords[:6]]
        
        # 提取速度
        speed_match = re.search(r'速度[：:]\s*(\d+)', llm_text) or \
                      re.search(r'speed\s*[=:]\s*(\d+)', llm_text, re.IGNORECASE)
        if speed_match:
            data['speed'] = int(speed_match.group(1))
        
        # 提取高度
        height_match = re.search(r'高度[：:]\s*(\d+(?:\.\d+)?)', llm_text) or \
                       re.search(r'height\s*[=:]\s*(\d+(?:\.\d+)?)', llm_text, re.IGNORECASE)
        if height_match:
            data['height'] = float(height_match.group(1))
        
        return self.parse_from_dict(data)


# ============================================================================
# 安全动作执行器主类
# ============================================================================

class SafeActionExecutor:
    """
    安全动作执行器主类
    
    主要功能：
    1. 接收高层指令并执行
    2. 安全检查（限位、碰撞检测）
    3. 异常处理与回滚
    4. 统一的仿真/真机接口
    """
    
    def __init__(self, 
                 simulation_mode: bool = True,
                 constraints: Optional[SafetyConstraints] = None,
                 port: str = None):
        """
        初始化安全动作执行器
        
        Args:
            simulation_mode: 是否使用仿真模式
            constraints: 安全约束配置
            port: 机械臂串口（真机模式使用）
        """
        self.simulation_mode = simulation_mode
        self.constraints = constraints or SafetyConstraints()
        
        # 组件初始化
        self.robot: Optional[RobotArmInterface] = None
        self.safety_checker = SafetyChecker(self.constraints)
        self.command_parser = CommandParser()
        self.action_history = ActionHistory()
        
        # 状态标志
        self._initialized = False
        self._is_executing = False
        self._emergency_stop_requested = False
        
        # 任务监控
        self._task_start_time: Optional[float] = None
        
        # 日志
        self.logger = logging.getLogger('SafeActionExecutor')
        self.logger.info(f"Initializing SafeActionExecutor in "
                        f"{'SIMULATION' if simulation_mode else 'REAL'} mode")
    
    def initialize(self) -> bool:
        """初始化执行器 - 连接机械臂"""
        if self._initialized:
            self.logger.warning("Already initialized")
            return True
        
        try:
            # 创建机械臂实例
            if self.simulation_mode:
                self.robot = SimulatedRobotArm()
            else:
                self.robot = MyCobotRobotArm(port=self.constraints.workspace_limits.get('port'))
            
            # 尝试连接
            if not self.robot.connect():
                if not self.simulation_mode:
                    self.logger.warning("Failed to connect to real robot, falling back to simulation")
                    self.robot = SimulatedRobotArm()
                    self.robot.connect()
                    self.simulation_mode = True
                else:
                    raise ConnectionError("Failed to initialize robot arm")
            
            self._initialized = True
            self.logger.info("SafeActionExecutor initialized successfully")
            self._log_initial_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self._initialized = False
            return False
    
    def _log_initial_state(self):
        """记录初始状态"""
        if self.robot:
            pose = self.robot.get_current_pose()
            joints = self.robot.get_current_joints()
            self.logger.info(f"Initial pose: {pose}")
            self.logger.info(f"Initial joints: {joints}")
            self.logger.info(f"Mode: {'SIMULATION' if self.simulation_mode else 'REAL'}")
    
    # ------------------------------------------------------------------------
    # 基础动作执行
    # ------------------------------------------------------------------------
    
    def _safe_move_to_pose(self, pose: Pose, speed: int = 30, 
                           record: Optional[ActionRecord] = None) -> bool:
        """
        安全移动到位姿（内部方法）
        
        包含完整的安全检查流程
        """
        if not self.robot or not self.robot.is_connected:
            self.logger.error("Robot not connected")
            return False
        
        # 获取当前状态
        current_pose = self.robot.get_current_pose()
        current_joints = self.robot.get_current_joints()
        
        # 注意：这里简化处理，实际需要正逆运动学
        # 仿真模式下直接使用目标关节
        target_joints = current_joints
        
        # 执行安全验证（失败时会抛出特定异常）
        validation = self.safety_checker.validate_move(
            current_pose, pose, current_joints, target_joints, speed
        )
        
        # 使用约束后的速度
        actual_speed = validation['constrained_speed']
        
        # 执行超时检查
        self._check_timeout()
        
        # 执行移动
        self.logger.info(f"Executing safe move to {pose} at speed {actual_speed}%")
        
        try:
            success = self.robot.move_to_pose(pose, actual_speed, wait=True)
            
            if success:
                new_pose = self.robot.get_current_pose()
                new_joints = self.robot.get_current_joints()
                self.logger.info(f"Move completed. New pose: {new_pose}")
                
                if record:
                    record.end_pose = new_pose
                    record.end_joints = new_joints
                
                return True
            else:
                raise ActionFailedException("Move command returned failure")
                
        except TimeoutException:
            self.logger.error("Move timed out")
            if record:
                record.success = False
            raise
            
        except Exception as e:
            self.logger.error(f"Move failed with exception: {e}")
            if record:
                record.success = False
            raise ActionFailedException(f"Move failed: {e}")
    
    def _safe_move_to_joints(self, joints: JointAngles, speed: int = 30,
                             record: Optional[ActionRecord] = None) -> bool:
        """安全移动到指定关节角度"""
        if not self.robot or not self.robot.is_connected:
            self.logger.error("Robot not connected")
            return False
        
        # 检查关节限位
        self.safety_checker.check_joint_limits(joints)
        
        # 约束速度
        actual_speed = self.safety_checker.check_speed(speed)
        
        # 超时检查
        self._check_timeout()
        
        self.logger.info(f"Executing safe joint move to {joints} at speed {actual_speed}%")
        
        try:
            success = self.robot.move_to_joints(joints, actual_speed, wait=True)
            
            if success:
                new_pose = self.robot.get_current_pose()
                new_joints = self.robot.get_current_joints()
                self.logger.info(f"Joint move completed. New joints: {new_joints}")
                
                if record:
                    record.end_pose = new_pose
                    record.end_joints = new_joints
                
                return True
            else:
                raise ActionFailedException("Joint move command returned failure")
                
        except Exception as e:
            self.logger.error(f"Joint move failed: {e}")
            if record:
                record.success = False
            raise ActionFailedException(f"Joint move failed: {e}")
    
    # ------------------------------------------------------------------------
    # 复合动作执行
    # ------------------------------------------------------------------------
    
    def _execute_pick(self, command: HighLevelCommand) -> bool:
        """
        执行抓取动作
        
        流程：
        1. 移动到安全高度
        2. 移动到目标上方
        3. 下降到抓取高度
        4. 激活吸泵
        5. 上升到安全高度
        """
        if not command.target_pose:
            raise ValueError("Pick action requires target_pose")
        
        target_pose = command.target_pose
        safe_height = command.safe_height
        pick_height = command.height or 90.0  # 默认抓取高度
        speed = command.speed
        
        self.logger.info(f"Executing PICK at {target_pose}, height={pick_height}mm")
        
        # 获取当前位姿
        current_pose = self.robot.get_current_pose()
        
        # 步骤1: 移动到目标上方安全高度
        safe_pose = Pose(
            x=target_pose.x, y=target_pose.y, z=safe_height,
            rx=target_pose.rx, ry=target_pose.ry, rz=target_pose.rz
        )
        
        # 记录整个PICK动作（用于回滚）
        pick_record = self.action_history.record_start(
            ActionType.PICK, current_pose, self.robot.get_current_joints()
        )
        
        try:
            # 移动到安全高度
            self.logger.info("Step 1/5: Moving to safe height above target")
            safe_record = self.action_history.record_start(
                ActionType.MOVE_TO_POSE,
                self.robot.get_current_pose(),
                self.robot.get_current_joints()
            )
            self._safe_move_to_pose(safe_pose, speed, safe_record)
            self.action_history.record_end(safe_record, True)
            
            # 步骤2: 下降到抓取高度
            self.logger.info("Step 2/5: Descending to pick height")
            pick_pose = Pose(
                x=target_pose.x, y=target_pose.y, z=pick_height,
                rx=target_pose.rx, ry=target_pose.ry, rz=target_pose.rz
            )
            descend_record = self.action_history.record_start(
                ActionType.MOVE_TO_POSE,
                self.robot.get_current_pose(),
                self.robot.get_current_joints()
            )
            self._safe_move_to_pose(pick_pose, speed, descend_record)
            self.action_history.record_end(descend_record, True)
            
            # 步骤3: 激活吸泵
            self.logger.info("Step 3/5: Activating gripper")
            self.robot.activate_gripper(True)
            time.sleep(0.5)  # 等待吸泵建立负压
            
            # 步骤4: 上升到安全高度
            self.logger.info("Step 4/5: Ascending to safe height with object")
            ascend_record = self.action_history.record_start(
                ActionType.MOVE_TO_POSE,
                self.robot.get_current_pose(),
                self.robot.get_current_joints()
            )
            self._safe_move_to_pose(safe_pose, speed, ascend_record)
            self.action_history.record_end(ascend_record, True)
            
            # 记录成功
            self.action_history.record_end(
                pick_record, True,
                end_pose=self.robot.get_current_pose(),
                end_joints=self.robot.get_current_joints()
            )
            
            self.logger.info("PICK action completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"PICK action failed: {e}")
            self.action_history.record_end(pick_record, False)
            
            # 尝试回滚
            self.logger.info("Attempting rollback after failed PICK")
            rollback_success = self._rollback_to_safe_state()
            
            if not rollback_success:
                self.logger.critical("Rollback failed! Robot may be in an unsafe state!")
            
            raise ActionFailedException(f"PICK failed: {e}")
    
    def _execute_place(self, command: HighLevelCommand) -> bool:
        """
        执行放置动作
        
        流程：
        1. 移动到目标上方安全高度
        2. 下降到放置高度
        3. 释放吸泵
        4. 上升到安全高度
        """
        if not command.target_pose:
            raise ValueError("Place action requires target_pose")
        
        target_pose = command.target_pose
        safe_height = command.safe_height
        place_height = command.height or 100.0
        speed = command.speed
        
        self.logger.info(f"Executing PLACE at {target_pose}, height={place_height}mm")
        
        current_pose = self.robot.get_current_pose()
        place_record = self.action_history.record_start(
            ActionType.PLACE, current_pose, self.robot.get_current_joints()
        )
        
        try:
            # 步骤1: 移动到目标上方安全高度
            self.logger.info("Step 1/4: Moving to safe height above target")
            safe_pose = Pose(
                x=target_pose.x, y=target_pose.y, z=safe_height,
                rx=target_pose.rx, ry=target_pose.ry, rz=target_pose.rz
            )
            self._safe_move_to_pose(safe_pose, speed)
            
            # 步骤2: 下降到放置高度
            self.logger.info("Step 2/4: Descending to place height")
            place_pose = Pose(
                x=target_pose.x, y=target_pose.y, z=place_height,
                rx=target_pose.rx, ry=target_pose.ry, rz=target_pose.rz
            )
            self._safe_move_to_pose(place_pose, speed)
            
            # 步骤3: 释放吸泵
            self.logger.info("Step 3/4: Deactivating gripper (releasing object)")
            self.robot.activate_gripper(False)
            time.sleep(0.5)
            
            # 步骤4: 上升到安全高度
            self.logger.info("Step 4/4: Ascending to safe height")
            self._safe_move_to_pose(safe_pose, speed)
            
            self.action_history.record_end(
                place_record, True,
                end_pose=self.robot.get_current_pose(),
                end_joints=self.robot.get_current_joints()
            )
            
            self.logger.info("PLACE action completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"PLACE action failed: {e}")
            self.action_history.record_end(place_record, False)
            
            self.logger.info("Attempting rollback after failed PLACE")
            self._rollback_to_safe_state()
            
            raise ActionFailedException(f"PLACE failed: {e}")
    
    def _execute_move_to_pose(self, command: HighLevelCommand) -> bool:
        """执行移动到位姿动作"""
        if not command.target_pose:
            raise ValueError("Move_to_pose action requires target_pose")
        
        current_pose = self.robot.get_current_pose()
        record = self.action_history.record_start(
            ActionType.MOVE_TO_POSE, current_pose, self.robot.get_current_joints()
        )
        
        try:
            success = self._safe_move_to_pose(command.target_pose, command.speed, record)
            self.action_history.record_end(
                record, success,
                end_pose=self.robot.get_current_pose(),
                end_joints=self.robot.get_current_joints()
            )
            return success
            
        except Exception as e:
            self.action_history.record_end(record, False)
            raise
    
    def _execute_move_joints(self, command: HighLevelCommand) -> bool:
        """执行移动关节动作"""
        if not command.target_joints:
            raise ValueError("Move_joints action requires target_joints")
        
        current_joints = self.robot.get_current_joints()
        record = self.action_history.record_start(
            ActionType.MOVE_JOINTS, self.robot.get_current_pose(), current_joints
        )
        
        try:
            success = self._safe_move_to_joints(command.target_joints, command.speed, record)
            self.action_history.record_end(
                record, success,
                end_pose=self.robot.get_current_pose(),
                end_joints=self.robot.get_current_joints()
            )
            return success
            
        except Exception as e:
            self.action_history.record_end(record, False)
            raise
    
    def _execute_home(self, command: HighLevelCommand) -> bool:
        """执行归零动作"""
        current_pose = self.robot.get_current_pose()
        record = self.action_history.record_start(
            ActionType.HOME, current_pose, self.robot.get_current_joints()
        )
        
        try:
            self.logger.info("Executing HOME (zero) position")
            home_joints = JointAngles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            success = self._safe_move_to_joints(home_joints, command.speed, record)
            
            self.action_history.record_end(
                record, success,
                end_pose=self.robot.get_current_pose(),
                end_joints=self.robot.get_current_joints()
            )
            
            self.logger.info("HOME action completed")
            return success
            
        except Exception as e:
            self.action_history.record_end(record, False)
            raise
    
    def _execute_relax(self, command: HighLevelCommand) -> bool:
        """执行放松动作"""
        current_pose = self.robot.get_current_pose()
        record = self.action_history.record_start(
            ActionType.RELAX, current_pose, self.robot.get_current_joints()
        )
        
        try:
            self.logger.info("Executing RELAX - releasing all joints")
            success = self.robot.relax()
            
            self.action_history.record_end(
                record, success,
                end_pose=self.robot.get_current_pose(),
                end_joints=self.robot.get_current_joints()
            )
            
            return success
            
        except Exception as e:
            self.action_history.record_end(record, False)
            raise
    
    # ------------------------------------------------------------------------
    # 公共执行接口
    # ------------------------------------------------------------------------
    
    def execute_command(self, command: HighLevelCommand) -> bool:
        """
        执行高层指令（主入口）
        
        Args:
            command: 高层指令对象
            
        Returns:
            bool: 是否执行成功
        """
        if not self._initialized:
            raise RuntimeError("Executor not initialized. Call initialize() first.")
        
        if self._is_executing:
            raise RuntimeError("Already executing a command")
        
        self._is_executing = True
        self._emergency_stop_requested = False
        self._task_start_time = time.time()
        
        try:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Executing command: {command}")
            self.logger.info(f"Start time: {datetime.fromtimestamp(self._task_start_time)}")
            self.logger.info(f"{'='*60}")
            
            # 根据动作类型分发执行
            action_map = {
                ActionType.MOVE_TO_POSE: self._execute_move_to_pose,
                ActionType.PICK: self._execute_pick,
                ActionType.PLACE: self._execute_place,
                ActionType.MOVE_JOINTS: self._execute_move_joints,
                ActionType.HOME: self._execute_home,
                ActionType.RELAX: self._execute_relax,
            }
            
            executor = action_map.get(command.action_type)
            if not executor:
                raise ValueError(f"Unknown action type: {command.action_type}")
            
            success = executor(command)
            
            # 记录总耗时
            total_time = time.time() - self._task_start_time
            status = "SUCCESS" if success else "FAILED"
            
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Command execution {status}")
            self.logger.info(f"Total time: {total_time:.2f} seconds")
            self.logger.info(f"{'='*60}")
            
            return success
            
        except SafetyException as e:
            self.logger.error(f"Safety exception during execution: {e}")
            # 安全异常已经处理过回滚
            raise
            
        except Exception as e:
            self.logger.error(f"Unexpected exception during execution: {e}")
            # 尝试回滚
            try:
                self._rollback_to_safe_state()
            except Exception as rollback_e:
                self.logger.critical(f"Rollback also failed: {rollback_e}")
            raise
            
        finally:
            self._is_executing = False
    
    def execute_from_dict(self, command_dict: Dict[str, Any]) -> bool:
        """从字典执行指令"""
        command = self.command_parser.parse_from_dict(command_dict)
        return self.execute_command(command)
    
    def execute_from_llm(self, llm_response: str) -> bool:
        """从大模型响应执行指令"""
        command = self.command_parser.parse_from_llm_response(llm_response)
        return self.execute_command(command)
    
    # ------------------------------------------------------------------------
    # 超时和紧急停止
    # ------------------------------------------------------------------------
    
    def _check_timeout(self):
        """检查是否超时"""
        if self._task_start_time is None:
            return
        
        elapsed = time.time() - self._task_start_time
        if elapsed > self.constraints.total_timeout:
            self.logger.error(f"Total task timeout after {elapsed:.1f}s")
            self.emergency_stop()
            raise TimeoutException(f"Task timed out after {elapsed:.1f}s")
    
    def emergency_stop(self):
        """紧急停止"""
        self.logger.warning("EMERGENCY STOP requested!")
        self._emergency_stop_requested = True
        
        if self.robot:
            self.robot.stop()
        
        self.logger.warning("Emergency stop executed")
    
    # ------------------------------------------------------------------------
    # 回滚机制
    # ------------------------------------------------------------------------
    
    def _rollback_to_safe_state(self) -> bool:
        """
        回滚到安全状态
        
        策略：
        1. 首先停止当前运动
        2. 释放吸泵（防止携带物体）
        3. 尝试回到最近的成功位置
        4. 如果无法回滚，尝试归零
        """
        self.logger.info("Starting rollback procedure...")
        
        try:
            # 步骤1: 立即停止
            self.logger.info("Rollback Step 1: Stopping all movement")
            if self.robot:
                self.robot.stop()
                self.robot.wait_for_stop(timeout=2.0)
            
            # 步骤2: 释放吸泵
            self.logger.info("Rollback Step 2: Releasing gripper")
            if self.robot:
                self.robot.activate_gripper(False)
            time.sleep(0.3)
            
            # 步骤3: 尝试回滚到最近的成功动作位置
            last_action = self.action_history.get_last_successful_action()
            if last_action and last_action.start_pose:
                self.logger.info(f"Rollback Step 3: Attempting to return to last known safe position")
                self.logger.info(f"  Last safe position: {last_action.start_pose}")
                
                try:
                    # 先回到安全高度
                    safe_pose = Pose(
                        x=last_action.start_pose.x,
                        y=last_action.start_pose.y,
                        z=self.constraints.default_safe_height,
                        rx=last_action.start_pose.rx,
                        ry=last_action.start_pose.ry,
                        rz=last_action.start_pose.rz
                    )
                    
                    if self.robot:
                        # 简化的回滚：只做基本移动，不做完整安全检查
                        self.robot.move_to_pose(safe_pose, speed=20, wait=True)
                        self.robot.move_to_pose(last_action.start_pose, speed=20, wait=True)
                    
                    self.logger.info("Rollback to last safe position completed")
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"Failed to rollback to last position: {e}")
            
            # 步骤4: 如果无法回滚到之前位置，尝试归零
            self.logger.info("Rollback Step 4: Attempting to go to HOME position")
            try:
                if self.robot:
                    self.robot.home(speed=20)
                self.logger.info("Rollback to HOME completed")
                return True
            except Exception as e:
                self.logger.error(f"Failed to go to HOME: {e}")
            
            self.logger.error("All rollback attempts failed")
            return False
            
        except Exception as e:
            self.logger.critical(f"Critical error during rollback: {e}")
            return False
    
    # ------------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------------
    
    def add_obstacle(self, pose: Pose, radius: float = 50.0, name: str = "obstacle"):
        """添加障碍物用于碰撞检测"""
        self.safety_checker.add_obstacle(pose, radius, name)
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        self.safety_checker.clear_obstacles()
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        state = {
            'initialized': self._initialized,
            'simulation_mode': self.simulation_mode,
            'is_executing': self._is_executing,
            'emergency_stop_requested': self._emergency_stop_requested,
        }
        
        if self.robot and self.robot.is_connected:
            state.update({
                'robot_connected': True,
                'current_pose': self.robot.get_current_pose().to_list(),
                'current_joints': self.robot.get_current_joints().to_list(),
                'is_moving': self.robot.is_moving,
                'gripper_activated': self.robot._gripper_activated,
            })
        else:
            state['robot_connected'] = False
        
        return state
    
    def shutdown(self):
        """关闭执行器"""
        self.logger.info("Shutting down SafeActionExecutor...")
        
        if self._is_executing:
            self.logger.warning("Shutting down while executing - stopping first")
            self.emergency_stop()
        
        if self.robot:
            try:
                # 释放吸泵
                self.robot.activate_gripper(False)
                # 断开连接
                self.robot.disconnect()
            except Exception as e:
                self.logger.warning(f"Error during robot disconnect: {e}")
        
        self._initialized = False
        self.logger.info("SafeActionExecutor shut down")


# ============================================================================
# 便捷函数
# ============================================================================

def create_executor(simulation: bool = True, **kwargs) -> SafeActionExecutor:
    """
    创建安全动作执行器的便捷函数
    
    Args:
        simulation: 是否使用仿真模式
        **kwargs: 其他传递给 SafeActionExecutor 的参数
    
    Returns:
        SafeActionExecutor: 已初始化的执行器
    """
    executor = SafeActionExecutor(simulation_mode=simulation, **kwargs)
    executor.initialize()
    return executor
