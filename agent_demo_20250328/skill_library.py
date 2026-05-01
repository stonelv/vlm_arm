"""
拖拽示教-可复用技能库
同济子豪兄 2024-5-23
支持轨迹录制、平滑处理、参数化抽象、保存检索、组合执行
"""

print('导入拖拽示教-可复用技能库模块')

import time
import os
import sys
import json
import math
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from copy import deepcopy

try:
    from pymycobot.mycobot import MyCobot
    from pymycobot import PI_PORT, PI_BAUD
    HAS_PYMYCOBOT = True
except ImportError:
    HAS_PYMYCOBOT = False
    print("警告: pymycobot 未安装，将使用模拟模式")

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("警告: RPi.GPIO 未安装，将使用模拟模式")


class EventType(Enum):
    PUMP_ON = "pump_on"
    PUMP_OFF = "pump_off"
    WAIT = "wait"


@dataclass
class TrajectoryPoint:
    timestamp: float
    angles: List[float]
    coords: Optional[List[float]] = None
    event: Optional[Dict] = None


@dataclass
class SkillParameter:
    name: str
    description: str
    param_type: str
    default_value: Any = None
    min_value: Any = None
    max_value: Any = None


@dataclass
class Skill:
    name: str
    description: str
    skill_type: str
    parameters: List[SkillParameter] = field(default_factory=list)
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    smoothed_trajectory: List[TrajectoryPoint] = field(default_factory=list)
    normalized_duration: float = 0.0
    metadata: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type,
            "parameters": [asdict(p) for p in self.parameters],
            "trajectory": [
                {
                    "timestamp": p.timestamp,
                    "angles": p.angles,
                    "coords": p.coords,
                    "event": p.event
                }
                for p in self.trajectory
            ],
            "smoothed_trajectory": [
                {
                    "timestamp": p.timestamp,
                    "angles": p.angles,
                    "coords": p.coords,
                    "event": p.event
                }
                for p in self.smoothed_trajectory
            ],
            "normalized_duration": self.normalized_duration,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Skill':
        trajectory = [
            TrajectoryPoint(
                timestamp=p["timestamp"],
                angles=p["angles"],
                coords=p.get("coords"),
                event=p.get("event")
            )
            for p in data.get("trajectory", [])
        ]
        smoothed_trajectory = [
            TrajectoryPoint(
                timestamp=p["timestamp"],
                angles=p["angles"],
                coords=p.get("coords"),
                event=p.get("event")
            )
            for p in data.get("smoothed_trajectory", [])
        ]
        parameters = [
            SkillParameter(**p) for p in data.get("parameters", [])
        ]
        return cls(
            name=data["name"],
            description=data["description"],
            skill_type=data["skill_type"],
            parameters=parameters,
            trajectory=trajectory,
            smoothed_trajectory=smoothed_trajectory,
            normalized_duration=data.get("normalized_duration", 0.0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time())
        )


@dataclass
class CompositeSkill(Skill):
    sub_skills: List[Tuple[str, Dict]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        data = super().to_dict()
        data["sub_skills"] = [
            {"skill_name": s[0], "params": s[1]} for s in self.sub_skills
        ]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CompositeSkill':
        skill = super().from_dict(data)
        skill.sub_skills = [
            (s["skill_name"], s["params"]) for s in data.get("sub_skills", [])
        ]
        return skill


class TrajectoryProcessor:
    @staticmethod
    def moving_average_smooth(points: List[TrajectoryPoint], window_size: int = 5) -> List[TrajectoryPoint]:
        if len(points) < window_size:
            return points
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(points)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(points), i + half_window + 1)
            
            window_points = points[start_idx:end_idx]
            window_angles = np.array([p.angles for p in window_points])
            
            avg_angles = np.mean(window_angles, axis=0).tolist()
            
            new_point = TrajectoryPoint(
                timestamp=points[i].timestamp,
                angles=avg_angles,
                coords=points[i].coords,
                event=points[i].event
            )
            smoothed.append(new_point)
        
        return smoothed
    
    @staticmethod
    def savitzky_golay_smooth(points: List[TrajectoryPoint], window_size: int = 7, polyorder: int = 3) -> List[TrajectoryPoint]:
        if len(points) < window_size:
            return points
        
        def savgol_coeffs(window_size, polyorder):
            half_window = (window_size - 1) // 2
            x = np.arange(-half_window, half_window + 1)
            X = np.vander(x, polyorder + 1)
            coeffs = np.linalg.pinv(X)[0]
            return coeffs
        
        coeffs = savgol_coeffs(window_size, polyorder)
        half_window = (window_size - 1) // 2
        
        smoothed = []
        angles_matrix = np.array([p.angles for p in points])
        
        for i in range(len(points)):
            if i < half_window or i >= len(points) - half_window:
                smoothed.append(points[i])
                continue
            
            start_idx = i - half_window
            end_idx = i + half_window + 1
            window_angles = angles_matrix[start_idx:end_idx]
            
            smoothed_angles = np.dot(coeffs, window_angles).tolist()
            
            new_point = TrajectoryPoint(
                timestamp=points[i].timestamp,
                angles=smoothed_angles,
                coords=points[i].coords,
                event=points[i].event
            )
            smoothed.append(new_point)
        
        return smoothed
    
    @staticmethod
    def time_normalize(points: List[TrajectoryPoint], target_duration: float = 5.0, num_points: int = 100) -> List[TrajectoryPoint]:
        if len(points) < 2:
            return points
        
        original_times = np.array([p.timestamp for p in points])
        original_times = original_times - original_times[0]
        original_duration = original_times[-1]
        
        if original_duration <= 0:
            return points
        
        target_times = np.linspace(0, target_duration, num_points)
        normalized_times = target_times / target_duration * original_duration
        
        angles_matrix = np.array([p.angles for p in points])
        
        normalized_points = []
        for i, t in enumerate(normalized_times):
            idx = np.searchsorted(original_times, t)
            if idx == 0:
                idx = 1
            elif idx >= len(original_times):
                idx = len(original_times) - 1
            
            t_prev = original_times[idx - 1]
            t_curr = original_times[idx]
            alpha = (t - t_prev) / (t_curr - t_prev + 1e-6)
            
            angles_prev = angles_matrix[idx - 1]
            angles_curr = angles_matrix[idx]
            interpolated_angles = (angles_prev * (1 - alpha) + angles_curr * alpha).tolist()
            
            coords = None
            event = None
            if points[idx].event:
                event = points[idx].event
            elif points[idx - 1].event:
                event = points[idx - 1].event
            
            new_point = TrajectoryPoint(
                timestamp=target_times[i],
                angles=interpolated_angles,
                coords=coords,
                event=event
            )
            normalized_points.append(new_point)
        
        return normalized_points
    
    @staticmethod
    def extract_keyframes(points: List[TrajectoryPoint], angle_threshold: float = 5.0) -> List[TrajectoryPoint]:
        if len(points) < 2:
            return points
        
        keyframes = [points[0]]
        
        for i in range(1, len(points)):
            prev_point = keyframes[-1]
            curr_point = points[i]
            
            angle_diff = np.abs(np.array(curr_point.angles) - np.array(prev_point.angles))
            max_diff = np.max(angle_diff)
            
            if max_diff >= angle_threshold or curr_point.event:
                keyframes.append(curr_point)
        
        if keyframes[-1] != points[-1]:
            keyframes.append(points[-1])
        
        return keyframes


class EnhancedRecorder:
    def __init__(self, mc=None, pump_controller=None):
        self.mc = mc
        self.pump_controller = pump_controller
        self.recording = False
        self.record_list: List[TrajectoryPoint] = []
        self.record_thread = None
        self.pump_state = False
        
        if HAS_GPIO and not pump_controller:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(20, GPIO.OUT)
            GPIO.setup(21, GPIO.OUT)
            GPIO.output(20, 1)
    
    def get_pump_state(self) -> bool:
        return self.pump_state
    
    def pump_on(self):
        if HAS_GPIO:
            GPIO.output(20, 0)
        self.pump_state = True
        print("吸泵开启")
    
    def pump_off(self):
        if HAS_GPIO:
            GPIO.output(20, 1)
            time.sleep(0.05)
            GPIO.output(21, 0)
            time.sleep(0.2)
            GPIO.output(21, 1)
        self.pump_state = False
        print("吸泵关闭")
    
    def start_recording(self, sample_interval: float = 0.1):
        self.record_list = []
        self.recording = True
        self.pump_state = False
        
        def _record():
            start_time = time.time()
            while self.recording:
                current_time = time.time() - start_time
                
                angles = None
                coords = None
                
                if HAS_PYMYCOBOT and self.mc:
                    try:
                        angles = self.mc.get_encoders()
                        if not angles:
                            angles = self.mc.get_angles()
                        if angles:
                            coords = self.mc.get_coords()
                    except Exception as e:
                        pass
                
                if angles is None:
                    angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                
                point = TrajectoryPoint(
                    timestamp=current_time,
                    angles=angles,
                    coords=coords,
                    event=None
                )
                self.record_list.append(point)
                
                print(f"\r录制中... 已录制 {len(self.record_list)} 个点, 时长: {current_time:.1f}s", end="")
                time.sleep(sample_interval)
        
        print("\n开始录制轨迹...")
        print("提示: 录制过程中可以手动操作吸泵，事件将被自动记录")
        self.record_thread = threading.Thread(target=_record, daemon=True)
        self.record_thread.start()
    
    def record_pump_event(self, event_type: EventType):
        if not self.recording:
            return
        
        current_time = time.time()
        if self.record_list:
            start_time = self.record_list[0].timestamp
            current_time -= start_time
        
        event_data = {
            "type": event_type.value,
            "timestamp": current_time
        }
        
        point = TrajectoryPoint(
            timestamp=current_time,
            angles=self.record_list[-1].angles if self.record_list else [0.0] * 6,
            coords=self.record_list[-1].coords if self.record_list else None,
            event=event_data
        )
        self.record_list.append(point)
        
        if event_type == EventType.PUMP_ON:
            self.pump_on()
        elif event_type == EventType.PUMP_OFF:
            self.pump_off()
        
        print(f"\n已记录事件: {event_type.value}")
    
    def stop_recording(self) -> List[TrajectoryPoint]:
        if self.recording:
            self.recording = False
            if self.record_thread:
                self.record_thread.join()
            print(f"\n停止录制，共录制 {len(self.record_list)} 个点")
        
        return self.record_list
    
    def create_skill_from_recording(self, name: str, description: str, 
                                     smooth: bool = True, normalize: bool = True,
                                     smooth_method: str = "moving_average",
                                     target_duration: float = 5.0) -> Skill:
        if not self.record_list:
            raise ValueError("没有录制的数据")
        
        trajectory = deepcopy(self.record_list)
        
        smoothed_trajectory = trajectory
        if smooth:
            if smooth_method == "moving_average":
                smoothed_trajectory = TrajectoryProcessor.moving_average_smooth(trajectory)
            elif smooth_method == "savitzky_golay":
                smoothed_trajectory = TrajectoryProcessor.savitzky_golay_smooth(trajectory)
        
        normalized_duration = target_duration
        if normalize:
            smoothed_trajectory = TrajectoryProcessor.time_normalize(
                smoothed_trajectory, target_duration=target_duration
            )
        
        start_angles = trajectory[0].angles
        end_angles = trajectory[-1].angles
        
        metadata = {
            "original_num_points": len(trajectory),
            "smoothed_num_points": len(smoothed_trajectory),
            "original_duration": trajectory[-1].timestamp - trajectory[0].timestamp if len(trajectory) > 1 else 0,
            "normalized_duration": normalized_duration,
            "start_angles": start_angles,
            "end_angles": end_angles,
            "smooth_method": smooth_method if smooth else None,
            "normalized": normalize
        }
        
        parameters = [
            SkillParameter(
                name="start_angles",
                description="起始关节角度",
                param_type="list",
                default_value=start_angles
            ),
            SkillParameter(
                name="end_angles",
                description="目标关节角度",
                param_type="list",
                default_value=end_angles
            ),
            SkillParameter(
                name="speed",
                description="运动速度比例 (0.1-2.0)",
                param_type="float",
                default_value=1.0,
                min_value=0.1,
                max_value=2.0
            ),
            SkillParameter(
                name="height_offset",
                description="高度偏移量",
                param_type="float",
                default_value=0.0
            )
        ]
        
        skill = Skill(
            name=name,
            description=description,
            skill_type="primitive",
            parameters=parameters,
            trajectory=trajectory,
            smoothed_trajectory=smoothed_trajectory,
            normalized_duration=normalized_duration,
            metadata=metadata
        )
        
        return skill


class SkillLibrary:
    def __init__(self, library_path: str = None):
        self.skills: Dict[str, Skill] = {}
        self.composite_skills: Dict[str, CompositeSkill] = {}
        
        if library_path is None:
            library_path = os.path.join(os.path.dirname(__file__), "skill_library")
        self.library_path = library_path
        
        os.makedirs(self.library_path, exist_ok=True)
        self._load_library()
    
    def _get_skill_path(self, name: str, is_composite: bool = False) -> str:
        subdir = "composite" if is_composite else "primitive"
        path = os.path.join(self.library_path, subdir)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"{name}.json")
    
    def _load_library(self):
        primitive_path = os.path.join(self.library_path, "primitive")
        if os.path.exists(primitive_path):
            for filename in os.listdir(primitive_path):
                if filename.endswith(".json"):
                    try:
                        filepath = os.path.join(primitive_path, filename)
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        skill = Skill.from_dict(data)
                        self.skills[skill.name] = skill
                    except Exception as e:
                        print(f"加载技能 {filename} 失败: {e}")
        
        composite_path = os.path.join(self.library_path, "composite")
        if os.path.exists(composite_path):
            for filename in os.listdir(composite_path):
                if filename.endswith(".json"):
                    try:
                        filepath = os.path.join(composite_path, filename)
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        skill = CompositeSkill.from_dict(data)
                        self.composite_skills[skill.name] = skill
                    except Exception as e:
                        print(f"加载复合技能 {filename} 失败: {e}")
        
        print(f"已加载 {len(self.skills)} 个基础技能, {len(self.composite_skills)} 个复合技能")
    
    def save_skill(self, skill: Skill):
        if isinstance(skill, CompositeSkill):
            filepath = self._get_skill_path(skill.name, is_composite=True)
            self.composite_skills[skill.name] = skill
        else:
            filepath = self._get_skill_path(skill.name, is_composite=False)
            self.skills[skill.name] = skill
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(skill.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"技能已保存: {skill.name} -> {filepath}")
    
    def load_skill(self, name: str) -> Optional[Skill]:
        if name in self.skills:
            return self.skills[name]
        if name in self.composite_skills:
            return self.composite_skills[name]
        
        filepath = self._get_skill_path(name, is_composite=False)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                skill = Skill.from_dict(data)
                self.skills[skill.name] = skill
                return skill
            except Exception as e:
                print(f"加载技能 {name} 失败: {e}")
        
        filepath = self._get_skill_path(name, is_composite=True)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                skill = CompositeSkill.from_dict(data)
                self.composite_skills[skill.name] = skill
                return skill
            except Exception as e:
                print(f"加载复合技能 {name} 失败: {e}")
        
        return None
    
    def delete_skill(self, name: str):
        if name in self.skills:
            del self.skills[name]
            filepath = self._get_skill_path(name, is_composite=False)
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"已删除技能: {name}")
        
        if name in self.composite_skills:
            del self.composite_skills[name]
            filepath = self._get_skill_path(name, is_composite=True)
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"已删除复合技能: {name}")
    
    def search_skills(self, keyword: str) -> List[Skill]:
        results = []
        keyword = keyword.lower()
        
        for name, skill in self.skills.items():
            if keyword in name.lower() or keyword in skill.description.lower():
                results.append(skill)
        
        for name, skill in self.composite_skills.items():
            if keyword in name.lower() or keyword in skill.description.lower():
                results.append(skill)
        
        return results
    
    def list_all_skills(self) -> Dict[str, List]:
        return {
            "primitive": list(self.skills.keys()),
            "composite": list(self.composite_skills.keys())
        }


class SkillExecutor:
    def __init__(self, mc=None, library: SkillLibrary = None):
        self.mc = mc
        self.library = library or SkillLibrary()
        self.executing = False
        
        if HAS_GPIO:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(20, GPIO.OUT)
            GPIO.setup(21, GPIO.OUT)
            GPIO.output(20, 1)
    
    def _apply_params_to_trajectory(self, trajectory: List[TrajectoryPoint], 
                                      params: Dict) -> List[TrajectoryPoint]:
        if not trajectory:
            return trajectory
        
        speed = params.get("speed", 1.0)
        start_angles = params.get("start_angles")
        end_angles = params.get("end_angles")
        height_offset = params.get("height_offset", 0.0)
        
        modified_trajectory = []
        
        original_start = np.array(trajectory[0].angles)
        original_end = np.array(trajectory[-1].angles)
        
        for point in trajectory:
            new_point = deepcopy(point)
            
            if start_angles is not None or end_angles is not None:
                progress = 0.0
                if len(trajectory) > 1:
                    total_duration = trajectory[-1].timestamp - trajectory[0].timestamp
                    if total_duration > 0:
                        progress = (point.timestamp - trajectory[0].timestamp) / total_duration
                
                current_angles = np.array(point.angles)
                
                if start_angles is not None:
                    start_offset = np.array(start_angles) - original_start
                    current_angles += start_offset * (1 - progress)
                
                if end_angles is not None:
                    end_offset = np.array(end_angles) - original_end
                    current_angles += end_offset * progress
                
                new_point.angles = current_angles.tolist()
            
            if speed != 1.0:
                new_point.timestamp = new_point.timestamp / speed
            
            modified_trajectory.append(new_point)
        
        return modified_trajectory
    
    def _execute_event(self, event: Dict):
        if not event:
            return
        
        event_type = event.get("type")
        
        if event_type == EventType.PUMP_ON.value:
            if HAS_GPIO:
                GPIO.output(20, 0)
            print("执行事件: 开启吸泵")
        
        elif event_type == EventType.PUMP_OFF.value:
            if HAS_GPIO:
                GPIO.output(20, 1)
                time.sleep(0.05)
                GPIO.output(21, 0)
                time.sleep(0.2)
                GPIO.output(21, 1)
            print("执行事件: 关闭吸泵")
        
        elif event_type == EventType.WAIT.value:
            duration = event.get("duration", 1.0)
            print(f"执行事件: 等待 {duration} 秒")
            time.sleep(duration)
    
    def execute_skill(self, skill: Skill, params: Dict = None,
                      use_smoothed: bool = True, speed: int = 80) -> bool:
        if params is None:
            params = {}
        
        self.executing = True
        
        try:
            trajectory = skill.smoothed_trajectory if use_smoothed and skill.smoothed_trajectory else skill.trajectory
            
            if not trajectory:
                print("错误: 技能没有轨迹数据")
                return False
            
            modified_trajectory = self._apply_params_to_trajectory(trajectory, params)
            
            print(f"\n开始执行技能: {skill.name}")
            print(f"参数: {params}")
            print(f"轨迹点数: {len(modified_trajectory)}")
            
            for i, point in enumerate(modified_trajectory):
                if not self.executing:
                    print("执行被中断")
                    return False
                
                if point.event:
                    self._execute_event(point.event)
                
                if HAS_PYMYCOBOT and self.mc:
                    try:
                        self.mc.set_encoders(point.angles, speed)
                    except Exception as e:
                        try:
                            self.mc.send_angles(point.angles, speed)
                        except Exception as e2:
                            print(f"运动控制错误: {e2}")
                
                if i < len(modified_trajectory) - 1:
                    next_point = modified_trajectory[i + 1]
                    time_diff = next_point.timestamp - point.timestamp
                    if time_diff > 0:
                        time.sleep(time_diff)
                
                print(f"\r执行进度: {i+1}/{len(modified_trajectory)}", end="")
            
            print(f"\n技能执行完成: {skill.name}")
            return True
        
        except Exception as e:
            print(f"执行技能时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.executing = False
    
    def execute_composite_skill(self, composite_skill: CompositeSkill, 
                                 global_params: Dict = None) -> bool:
        if global_params is None:
            global_params = {}
        
        print(f"\n开始执行复合技能: {composite_skill.name}")
        print(f"包含 {len(composite_skill.sub_skills)} 个子技能")
        
        for i, (skill_name, local_params) in enumerate(composite_skill.sub_skills):
            if not self.executing:
                print("复合技能执行被中断")
                return False
            
            merged_params = {**global_params, **local_params}
            
            skill = self.library.load_skill(skill_name)
            if not skill:
                print(f"警告: 找不到子技能 {skill_name}，跳过")
                continue
            
            print(f"\n[{i+1}/{len(composite_skill.sub_skills)}] 执行子技能: {skill_name}")
            success = self.execute_skill(skill, merged_params)
            
            if not success:
                print(f"子技能 {skill_name} 执行失败，复合技能中止")
                return False
        
        print(f"\n复合技能执行完成: {composite_skill.name}")
        return True
    
    def stop_execution(self):
        self.executing = False
        print("停止执行")


class SkillComposer:
    def __init__(self, library: SkillLibrary = None):
        self.library = library or SkillLibrary()
    
    def create_composite_skill(self, name: str, description: str,
                                 sub_skills: List[Tuple[str, Dict]]) -> CompositeSkill:
        composite = CompositeSkill(
            name=name,
            description=description,
            skill_type="composite",
            sub_skills=sub_skills,
            parameters=[
                SkillParameter(
                    name="global_speed",
                    description="全局速度比例",
                    param_type="float",
                    default_value=1.0
                )
            ]
        )
        
        return composite
    
    def create_pick_place_composite(self, 
                                     pick_skill_name: str = "pick_up",
                                     move_skill_name: str = "move",
                                     place_skill_name: str = "place_down",
                                     composite_name: str = "pick_and_place") -> CompositeSkill:
        sub_skills = [
            (pick_skill_name, {"height_offset": 0.0}),
            (move_skill_name, {"speed": 0.8}),
            (place_skill_name, {"height_offset": 0.0})
        ]
        
        composite = self.create_composite_skill(
            name=composite_name,
            description="拿起-移动-放下复合技能: 执行拾取、移动、放置的完整操作流程",
            sub_skills=sub_skills
        )
        
        return composite


def create_demo_skills(library: SkillLibrary):
    print("\n=== 创建演示技能 ===")
    
    pick_trajectory = [
        TrajectoryPoint(timestamp=0.0, angles=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], coords=[200, 0, 200, 0, 180, 90]),
        TrajectoryPoint(timestamp=1.0, angles=[0.0, -30.0, 30.0, 0.0, 0.0, 0.0], coords=[200, 0, 150, 0, 180, 90]),
        TrajectoryPoint(timestamp=2.0, angles=[0.0, -50.0, 50.0, 0.0, 0.0, 0.0], coords=[200, 0, 100, 0, 180, 90]),
        TrajectoryPoint(timestamp=2.1, angles=[0.0, -50.0, 50.0, 0.0, 0.0, 0.0], coords=[200, 0, 100, 0, 180, 90],
                        event={"type": "pump_on", "timestamp": 2.1}),
        TrajectoryPoint(timestamp=3.0, angles=[0.0, -30.0, 30.0, 0.0, 0.0, 0.0], coords=[200, 0, 150, 0, 180, 90]),
        TrajectoryPoint(timestamp=4.0, angles=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], coords=[200, 0, 200, 0, 180, 90]),
    ]
    
    pick_skill = Skill(
        name="pick_up",
        description="拾取技能: 下降到物体位置，开启吸泵，然后上升",
        skill_type="primitive",
        trajectory=pick_trajectory,
        smoothed_trajectory=pick_trajectory,
        normalized_duration=4.0,
        parameters=[
            SkillParameter(name="pick_xy", description="拾取位置XY坐标", param_type="list", default_value=[200, 0]),
            SkillParameter(name="pick_height", description="拾取高度", param_type="float", default_value=100.0),
            SkillParameter(name="safe_height", description="安全高度", param_type="float", default_value=200.0),
            SkillParameter(name="speed", description="速度", param_type="float", default_value=1.0),
        ],
        metadata={"demo": True}
    )
    
    move_trajectory = [
        TrajectoryPoint(timestamp=0.0, angles=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], coords=[200, 0, 200, 0, 180, 90]),
        TrajectoryPoint(timestamp=1.0, angles=[30.0, 0.0, 0.0, 0.0, 0.0, 0.0], coords=[150, 100, 200, 0, 180, 90]),
        TrajectoryPoint(timestamp=2.0, angles=[60.0, 0.0, 0.0, 0.0, 0.0, 0.0], coords=[100, 180, 200, 0, 180, 90]),
    ]
    
    move_skill = Skill(
        name="move",
        description="移动技能: 在安全高度水平移动到目标位置",
        skill_type="primitive",
        trajectory=move_trajectory,
        smoothed_trajectory=move_trajectory,
        normalized_duration=2.0,
        parameters=[
            SkillParameter(name="start_xy", description="起始XY坐标", param_type="list", default_value=[200, 0]),
            SkillParameter(name="target_xy", description="目标XY坐标", param_type="list", default_value=[100, 180]),
            SkillParameter(name="height", description="移动高度", param_type="float", default_value=200.0),
            SkillParameter(name="speed", description="速度", param_type="float", default_value=1.0),
        ],
        metadata={"demo": True}
    )
    
    place_trajectory = [
        TrajectoryPoint(timestamp=0.0, angles=[60.0, 0.0, 0.0, 0.0, 0.0, 0.0], coords=[100, 180, 200, 0, 180, 90]),
        TrajectoryPoint(timestamp=1.0, angles=[60.0, -30.0, 30.0, 0.0, 0.0, 0.0], coords=[100, 180, 150, 0, 180, 90]),
        TrajectoryPoint(timestamp=2.0, angles=[60.0, -50.0, 50.0, 0.0, 0.0, 0.0], coords=[100, 180, 100, 0, 180, 90]),
        TrajectoryPoint(timestamp=2.1, angles=[60.0, -50.0, 50.0, 0.0, 0.0, 0.0], coords=[100, 180, 100, 0, 180, 90],
                        event={"type": "pump_off", "timestamp": 2.1}),
        TrajectoryPoint(timestamp=3.0, angles=[60.0, -30.0, 30.0, 0.0, 0.0, 0.0], coords=[100, 180, 150, 0, 180, 90]),
        TrajectoryPoint(timestamp=4.0, angles=[60.0, 0.0, 0.0, 0.0, 0.0, 0.0], coords=[100, 180, 200, 0, 180, 90]),
    ]
    
    place_skill = Skill(
        name="place_down",
        description="放置技能: 下降到放置位置，关闭吸泵，然后上升",
        skill_type="primitive",
        trajectory=place_trajectory,
        smoothed_trajectory=place_trajectory,
        normalized_duration=4.0,
        parameters=[
            SkillParameter(name="place_xy", description="放置位置XY坐标", param_type="list", default_value=[100, 180]),
            SkillParameter(name="place_height", description="放置高度", param_type="float", default_value=100.0),
            SkillParameter(name="safe_height", description="安全高度", param_type="float", default_value=200.0),
            SkillParameter(name="speed", description="速度", param_type="float", default_value=1.0),
        ],
        metadata={"demo": True}
    )
    
    library.save_skill(pick_skill)
    library.save_skill(move_skill)
    library.save_skill(place_skill)
    
    print("已创建演示技能: pick_up, move, place_down")
    
    composer = SkillComposer(library)
    composite_skill = composer.create_pick_place_composite(
        pick_skill_name="pick_up",
        move_skill_name="move",
        place_skill_name="place_down",
        composite_name="pick_and_place"
    )
    
    library.save_skill(composite_skill)
    print("已创建复合技能: pick_and_place")
    
    return pick_skill, move_skill, place_skill, composite_skill


def run_interactive_demo():
    print("\n" + "="*60)
    print("拖拽示教-可复用技能库 交互式演示")
    print("="*60)
    
    library = SkillLibrary()
    
    if not library.skills:
        create_demo_skills(library)
    
    executor = SkillExecutor(library=library)
    composer = SkillComposer(library=library)
    
    while True:
        print("\n" + "-"*40)
        print("菜单:")
        print("1. 列出所有技能")
        print("2. 搜索技能")
        print("3. 查看技能详情")
        print("4. 执行技能 (模拟)")
        print("5. 创建复合技能示例")
        print("6. 执行复合技能 (模拟)")
        print("q. 退出")
        print("-"*40)
        
        choice = input("\n请选择操作: ").strip().lower()
        
        if choice == "q":
            print("再见!")
            break
        
        elif choice == "1":
            skills = library.list_all_skills()
            print("\n基础技能:")
            for name in skills["primitive"]:
                print(f"  - {name}")
            print("\n复合技能:")
            for name in skills["composite"]:
                print(f"  - {name}")
        
        elif choice == "2":
            keyword = input("请输入搜索关键词: ").strip()
            results = library.search_skills(keyword)
            print(f"\n找到 {len(results)} 个相关技能:")
            for skill in results:
                print(f"  - {skill.name}: {skill.description}")
        
        elif choice == "3":
            name = input("请输入技能名称: ").strip()
            skill = library.load_skill(name)
            if skill:
                print(f"\n技能详情: {skill.name}")
                print(f"  描述: {skill.description}")
                print(f"  类型: {skill.skill_type}")
                print(f"  轨迹点数: {len(skill.trajectory)}")
                print(f"  标准化时长: {skill.normalized_duration}s")
                print(f"  参数:")
                for param in skill.parameters:
                    print(f"    - {param.name}: {param.description} (默认: {param.default_value})")
                if isinstance(skill, CompositeSkill):
                    print(f"  子技能:")
                    for sub_name, params in skill.sub_skills:
                        print(f"    - {sub_name}: {params}")
            else:
                print(f"找不到技能: {name}")
        
        elif choice == "4":
            name = input("请输入要执行的技能名称: ").strip()
            skill = library.load_skill(name)
            if skill:
                print(f"\n模拟执行技能: {name}")
                print("注意: 这是模拟模式，实际运动需要连接机械臂")
                executor.execute_skill(skill, use_smoothed=True)
            else:
                print(f"找不到技能: {name}")
        
        elif choice == "5":
            print("\n创建复合技能示例: pick_and_place_custom")
            print("将 pick_up -> move -> place_down 组合成新技能")
            
            composite = composer.create_composite_skill(
                name="pick_and_place_custom",
                description="自定义的拿起-移动-放下复合技能",
                sub_skills=[
                    ("pick_up", {"speed": 0.8}),
                    ("move", {"speed": 1.0}),
                    ("place_down", {"speed": 0.8})
                ]
            )
            library.save_skill(composite)
            print(f"已创建并保存复合技能: {composite.name}")
        
        elif choice == "6":
            name = input("请输入复合技能名称 (默认: pick_and_place): ").strip()
            if not name:
                name = "pick_and_place"
            
            skill = library.load_skill(name)
            if skill and isinstance(skill, CompositeSkill):
                print(f"\n模拟执行复合技能: {name}")
                executor.execute_composite_skill(skill)
            else:
                print(f"不是复合技能或找不到: {name}")
        
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    run_interactive_demo()
