#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全禁区模块 - 实时视觉分析与碰撞避让
同济子豪兄 & AI Assistant 2024-2026

功能：
1. 定义三维空间中的"禁止进入"区域（球形、矩形等）
2. 使用多模态视觉模型实时分析机械臂工作画面
3. 检测机械臂末端或抓取目标是否接近禁区
4. 自动规划绕行路径或发出预警
5. 与现有坐标动作指令体系兼容
"""

import time
import json
import base64
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Callable, Dict, Any
import threading

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: cv2 未安装，RealTimeVisualAnalyzer 将不可用")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai 未安装，VLM 实时检测将不可用")


VLM_PROMPT_DETECT_OBJECTS = '''
你是一个机械臂视觉检测系统。请分析这张图片，检测以下目标对象：
1. 机械臂末端（end_effector）- 吸泵或夹具所在位置
2. 待抓取目标（target_object）- 需要移动的物体

对于每个检测到的对象，请输出以下JSON格式：
{
    "objects": [
        {
            "name": "end_effector",
            "bbox_xyxy": [[x1, y1], [x2, y2]],
            "confidence": 0.95
        },
        {
            "name": "target_object",
            "bbox_xyxy": [[x1, y1], [x2, y2]],
            "confidence": 0.90
        }
    ]
}

注意：
- bbox_xyxy 是对象的左上角和右下角像素坐标，值范围 0-999（相对于图像宽高）
- 如果某个对象未检测到，可以省略该对象或设置 confidence 为 0
- 只输出JSON，不要输出其他内容
'''


@dataclass
class VLMConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o"
    max_tokens: int = 1000
    temperature: float = 0.0


class ZoneType(Enum):
    SPHERE = "sphere"
    RECTANGLE = "rectangle"
    CYLINDER = "cylinder"


class AlertLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


@dataclass
class SafetyZone:
    name: str
    zone_type: ZoneType
    center: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    warning_distance: float = 20.0
    danger_distance: float = 10.0
    enabled: bool = True
    color: Tuple[int, int, int] = (0, 0, 255)

    def is_inside(self, point: Tuple[float, float, float]) -> bool:
        if not self.enabled:
            return False
        
        if self.zone_type == ZoneType.SPHERE:
            radius = self.dimensions[0]
            distance = self._distance_to_center(point)
            return distance <= radius
        
        elif self.zone_type == ZoneType.RECTANGLE:
            x, y, z = point
            cx, cy, cz = self.center
            dx, dy, dz = self.dimensions
            return (cx - dx/2 <= x <= cx + dx/2 and
                    cy - dy/2 <= y <= cy + dy/2 and
                    cz - dz/2 <= z <= cz + dz/2)
        
        elif self.zone_type == ZoneType.CYLINDER:
            radius = self.dimensions[0]
            height = self.dimensions[1]
            cx, cy, cz = self.center
            x, y, z = point
            horizontal_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            vertical_dist = abs(z - cz)
            return horizontal_dist <= radius and vertical_dist <= height/2
        
        return False

    def distance_to_zone(self, point: Tuple[float, float, float]) -> float:
        if not self.enabled:
            return float('inf')
        
        if self.is_inside(point):
            return 0.0
        
        if self.zone_type == ZoneType.SPHERE:
            radius = self.dimensions[0]
            distance = self._distance_to_center(point)
            return max(0, distance - radius)
        
        elif self.zone_type == ZoneType.RECTANGLE:
            x, y, z = point
            cx, cy, cz = self.center
            dx, dy, dz = self.dimensions
            
            closest_x = np.clip(x, cx - dx/2, cx + dx/2)
            closest_y = np.clip(y, cy - dy/2, cy + dy/2)
            closest_z = np.clip(z, cz - dz/2, cz + dz/2)
            
            return np.sqrt((x - closest_x)**2 + (y - closest_y)**2 + (z - closest_z)**2)
        
        elif self.zone_type == ZoneType.CYLINDER:
            radius = self.dimensions[0]
            height = self.dimensions[1]
            cx, cy, cz = self.center
            x, y, z = point
            
            horizontal_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            horizontal_dist = max(0, horizontal_dist - radius)
            
            vertical_dist = abs(z - cz) - height/2
            vertical_dist = max(0, vertical_dist)
            
            return np.sqrt(horizontal_dist**2 + vertical_dist**2)
        
        return float('inf')

    def get_alert_level(self, point: Tuple[float, float, float]) -> AlertLevel:
        if not self.enabled:
            return AlertLevel.SAFE
        
        if self.is_inside(point):
            return AlertLevel.EMERGENCY
        
        distance = self.distance_to_zone(point)
        
        if distance <= self.danger_distance:
            return AlertLevel.DANGER
        elif distance <= self.warning_distance:
            return AlertLevel.WARNING
        else:
            return AlertLevel.SAFE

    def _distance_to_center(self, point: Tuple[float, float, float]) -> float:
        cx, cy, cz = self.center
        x, y, z = point
        return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)


@dataclass
class ObjectState:
    name: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0, 0, 0)
    timestamp: float = 0.0
    confidence: float = 1.0


class ZoneManager:
    def __init__(self):
        self.zones: List[SafetyZone] = []
        self._lock = threading.Lock()

    def add_zone(self, zone: SafetyZone) -> None:
        with self._lock:
            self.zones.append(zone)

    def remove_zone(self, name: str) -> bool:
        with self._lock:
            for i, zone in enumerate(self.zones):
                if zone.name == name:
                    self.zones.pop(i)
                    return True
            return False

    def get_zone(self, name: str) -> Optional[SafetyZone]:
        with self._lock:
            for zone in self.zones:
                if zone.name == name:
                    return zone
            return None

    def update_zone(self, name: str, **kwargs) -> bool:
        with self._lock:
            zone = self.get_zone(name)
            if zone:
                for key, value in kwargs.items():
                    if hasattr(zone, key):
                        setattr(zone, key, value)
                return True
            return False

    def list_zones(self) -> List[SafetyZone]:
        with self._lock:
            return list(self.zones)

    def check_all_zones(self, point: Tuple[float, float, float]) -> Tuple[AlertLevel, List[Tuple[SafetyZone, AlertLevel]]]:
        highest_level = AlertLevel.SAFE
        zone_alerts = []
        
        with self._lock:
            for zone in self.zones:
                alert_level = zone.get_alert_level(point)
                zone_alerts.append((zone, alert_level))
                
                if self._alert_level_priority(alert_level) > self._alert_level_priority(highest_level):
                    highest_level = alert_level
        
        return highest_level, zone_alerts

    def get_closest_zone(self, point: Tuple[float, float, float]) -> Tuple[Optional[SafetyZone], float]:
        closest_zone = None
        min_distance = float('inf')
        
        with self._lock:
            for zone in self.zones:
                distance = zone.distance_to_zone(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_zone = zone
        
        return closest_zone, min_distance

    def _alert_level_priority(self, level: AlertLevel) -> int:
        priority = {
            AlertLevel.SAFE: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.DANGER: 2,
            AlertLevel.EMERGENCY: 3
        }
        return priority.get(level, 0)


class PathPlanner:
    def __init__(self, zone_manager: ZoneManager):
        self.zone_manager = zone_manager
        self.safe_height = 220.0
        self.min_detour_distance = 60.0
        self.arc_segments = 8

    def check_path_safe(self, start: Tuple[float, float, float], 
                         end: Tuple[float, float, float],
                         steps: int = 20) -> Tuple[bool, List[Tuple[float, AlertLevel]]]:
        path_alerts = []
        is_safe = True
        
        for i in range(steps + 1):
            alpha = i / steps
            point = (
                start[0] + alpha * (end[0] - start[0]),
                start[1] + alpha * (end[1] - start[1]),
                start[2] + alpha * (end[2] - start[2])
            )
            
            alert_level, _ = self.zone_manager.check_all_zones(point)
            path_alerts.append((alpha, alert_level))
            
            if alert_level in [AlertLevel.DANGER, AlertLevel.EMERGENCY]:
                is_safe = False
        
        return is_safe, path_alerts

    def _get_obstacle_info(self, start: Tuple[float, float, float], 
                            end: Tuple[float, float, float]) -> List[Tuple[SafetyZone, float]]:
        obstacles = []
        mid_point = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            (start[2] + end[2]) / 2
        )
        
        for zone in self.zone_manager.list_zones():
            if not zone.enabled:
                continue
            
            dist_to_start = zone.distance_to_zone(start)
            dist_to_end = zone.distance_to_zone(end)
            dist_to_mid = zone.distance_to_zone(mid_point)
            
            if dist_to_mid < 100 or dist_to_start < 50 or dist_to_end < 50:
                obstacles.append((zone, dist_to_mid))
        
        obstacles.sort(key=lambda x: x[1])
        return obstacles

    def _generate_arc_waypoints(self, start: Tuple[float, float, float],
                                 end: Tuple[float, float, float],
                                 obstacle_center: Tuple[float, float, float],
                                 obstacle_radius: float,
                                 clockwise: bool = True) -> List[Tuple[float, float, float]]:
        sx, sy, sz = start
        ex, ey, ez = end
        ox, oy, oz = obstacle_center
        
        safe_radius = obstacle_radius + self.min_detour_distance
        
        vec_start = np.array([sx - ox, sy - oy])
        vec_end = np.array([ex - ox, ey - oy])
        
        dist_start = np.linalg.norm(vec_start)
        dist_end = np.linalg.norm(vec_end)
        
        if dist_start < 1e-6 or dist_end < 1e-6:
            return [start, end]
        
        angle_start = np.arctan2(vec_start[1], vec_start[0])
        angle_end = np.arctan2(vec_end[1], vec_end[0])
        
        tangent_offset = np.arcsin(safe_radius / max(dist_start, safe_radius + 1e-6))
        
        if clockwise:
            arc_start_angle = angle_start - tangent_offset
            arc_end_angle = angle_end + tangent_offset
            
            while arc_end_angle > arc_start_angle:
                arc_end_angle -= 2 * np.pi
        else:
            arc_start_angle = angle_start + tangent_offset
            arc_end_angle = angle_end - tangent_offset
            
            while arc_end_angle < arc_start_angle:
                arc_end_angle += 2 * np.pi
        
        waypoints = [start]
        
        if dist_start > safe_radius:
            enter_x = ox + safe_radius * np.cos(arc_start_angle)
            enter_y = oy + safe_radius * np.sin(arc_start_angle)
            enter_point = (float(enter_x), float(enter_y), sz)
            waypoints.append(enter_point)
        
        num_arc_points = max(self.arc_segments, 3)
        for i in range(1, num_arc_points):
            alpha = i / num_arc_points
            angle = arc_start_angle + alpha * (arc_end_angle - arc_start_angle)
            
            x = ox + safe_radius * np.cos(angle)
            y = oy + safe_radius * np.sin(angle)
            z = sz + alpha * (ez - sz)
            
            waypoints.append((float(x), float(y), float(z)))
        
        if dist_end > safe_radius:
            exit_x = ox + safe_radius * np.cos(arc_end_angle)
            exit_y = oy + safe_radius * np.sin(arc_end_angle)
            exit_point = (float(exit_x), float(exit_y), ez)
            waypoints.append(exit_point)
        
        waypoints.append(end)
        
        return waypoints

    def plan_alternate_path(self, start: Tuple[float, float, float],
                             end: Tuple[float, float, float],
                             current_z: float = None) -> List[Tuple[float, float, float]]:
        if current_z is None:
            current_z = start[2] if start[2] > 0 else self.safe_height
        
        is_safe, _ = self.check_path_safe(start, end)
        if is_safe:
            return [start, end]
        
        obstacles = self._get_obstacle_info(start, end)
        
        if obstacles:
            primary_zone, _ = obstacles[0]
            ox, oy, oz = primary_zone.center
            
            if primary_zone.zone_type == ZoneType.SPHERE:
                obstacle_radius = primary_zone.dimensions[0]
            else:
                obstacle_radius = max(primary_zone.dimensions[0], primary_zone.dimensions[1]) / 2
            
            clockwise_waypoints = self._generate_arc_waypoints(
                start, end, (ox, oy, oz), obstacle_radius, clockwise=True
            )
            
            all_safe = True
            for i in range(len(clockwise_waypoints) - 1):
                safe, _ = self.check_path_safe(clockwise_waypoints[i], clockwise_waypoints[i + 1])
                if not safe:
                    all_safe = False
                    break
            
            if all_safe:
                return clockwise_waypoints
            
            counter_clockwise_waypoints = self._generate_arc_waypoints(
                start, end, (ox, oy, oz), obstacle_radius, clockwise=False
            )
            
            all_safe = True
            for i in range(len(counter_clockwise_waypoints) - 1):
                safe, _ = self.check_path_safe(counter_clockwise_waypoints[i], counter_clockwise_waypoints[i + 1])
                if not safe:
                    all_safe = False
                    break
            
            if all_safe:
                return counter_clockwise_waypoints
        
        start_high = (start[0], start[1], self.safe_height)
        end_high = (end[0], end[1], self.safe_height)
        
        waypoints = []
        
        if start[2] < self.safe_height:
            safe, _ = self.check_path_safe(start, start_high)
            if safe:
                waypoints.append(start)
                waypoints.append(start_high)
            else:
                waypoints.append(start)
        else:
            waypoints.append(start)
        
        safe, _ = self.check_path_safe(start_high, end_high)
        if safe:
            waypoints.append(end_high)
        else:
            zone, _ = self.zone_manager.get_closest_zone(start_high)
            if zone:
                cx, cy, cz = zone.center
                radius = zone.dimensions[0] if zone.zone_type == ZoneType.SPHERE else max(zone.dimensions)
                
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                
                dx = mid_x - cx
                dy = mid_y - cy
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    scale = (radius + self.min_detour_distance) / dist
                    detour_x = cx + dx * scale
                    detour_y = cy + dy * scale
                    
                    detour_point = (detour_x, detour_y, self.safe_height)
                    
                    if self.check_path_safe(start_high, detour_point)[0]:
                        waypoints.append(detour_point)
        
        waypoints.append(end_high)
        
        if end[2] < self.safe_height:
            safe, _ = self.check_path_safe(end_high, end)
            if safe:
                waypoints.append(end)
        else:
            waypoints.append(end)
        
        if waypoints[0] != start:
            waypoints.insert(0, start)
        
        if waypoints[-1] != end:
            waypoints.append(end)
        
        return waypoints

    def plan_smooth_path(self, start: Tuple[float, float, float],
                          end: Tuple[float, float, float],
                          num_waypoints: int = 10) -> List[Tuple[float, float, float]]:
        is_safe, _ = self.check_path_safe(start, end)
        if is_safe:
            return [start, end]
        
        waypoints = self.plan_alternate_path(start, end)
        
        if len(waypoints) <= 2:
            return waypoints
        
        smooth_waypoints = [waypoints[0]]
        
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            
            for j in range(1, num_waypoints):
                alpha = j / num_waypoints
                point = (
                    p1[0] + alpha * (p2[0] - p1[0]),
                    p1[1] + alpha * (p2[1] - p1[1]),
                    p1[2] + alpha * (p2[2] - p1[2])
                )
                smooth_waypoints.append(point)
        
        smooth_waypoints.append(waypoints[-1])
        
        return smooth_waypoints

    def is_point_safe(self, point: Tuple[float, float, float]) -> bool:
        alert_level, _ = self.zone_manager.check_all_zones(point)
        return alert_level == AlertLevel.SAFE

    def estimate_path_length(self, waypoints: List[Tuple[float, float, float]]) -> float:
        total_length = 0.0
        for i in range(len(waypoints) - 1):
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i + 1])
            total_length += np.linalg.norm(p2 - p1)
        return total_length


class SafetyMonitor:
    def __init__(self, zone_manager: ZoneManager, path_planner: PathPlanner):
        self.zone_manager = zone_manager
        self.path_planner = path_planner
        self.current_end_effector: Optional[ObjectState] = None
        self.current_target: Optional[ObjectState] = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable] = []
        self._last_update = 0.0

    def add_alert_callback(self, callback: Callable[[AlertLevel, str, ObjectState], None]) -> None:
        self._alert_callbacks.append(callback)

    def update_end_effector(self, position: Tuple[float, float, float],
                             velocity: Tuple[float, float, float] = (0, 0, 0),
                             confidence: float = 1.0) -> None:
        self.current_end_effector = ObjectState(
            name="end_effector",
            position=position,
            velocity=velocity,
            timestamp=time.time(),
            confidence=confidence
        )
        self._check_alerts(self.current_end_effector)

    def update_target(self, position: Tuple[float, float, float],
                      velocity: Tuple[float, float, float] = (0, 0, 0),
                      confidence: float = 1.0) -> None:
        self.current_target = ObjectState(
            name="grasp_target",
            position=position,
            velocity=velocity,
            timestamp=time.time(),
            confidence=confidence
        )
        self._check_alerts(self.current_target)

    def _check_alerts(self, obj: ObjectState) -> None:
        alert_level, zone_alerts = self.zone_manager.check_all_zones(obj.position)
        
        if alert_level != AlertLevel.SAFE:
            for zone, level in zone_alerts:
                if level != AlertLevel.SAFE:
                    self._trigger_alert(level, zone.name, obj)

    def _trigger_alert(self, level: AlertLevel, zone_name: str, obj: ObjectState) -> None:
        for callback in self._alert_callbacks:
            try:
                callback(level, zone_name, obj)
            except Exception as e:
                print(f"Alert callback error: {e}")

    def start_monitoring(self, update_interval: float = 0.1) -> None:
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(update_interval,),
            daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def _monitor_loop(self, interval: float) -> None:
        while self._monitoring:
            try:
                if self.current_end_effector:
                    self._predict_future_collision(self.current_end_effector)
                if self.current_target:
                    self._predict_future_collision(self.current_target)
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(interval)

    def _predict_future_collision(self, obj: ObjectState, prediction_steps: int = 5) -> None:
        if obj.velocity == (0, 0, 0):
            return
        
        dt = 0.1
        for step in range(1, prediction_steps + 1):
            future_pos = (
                obj.position[0] + obj.velocity[0] * step * dt,
                obj.position[1] + obj.velocity[1] * step * dt,
                obj.position[2] + obj.velocity[2] * step * dt
            )
            
            alert_level, _ = self.zone_manager.check_all_zones(future_pos)
            
            if alert_level in [AlertLevel.DANGER, AlertLevel.EMERGENCY]:
                self._trigger_alert(alert_level, f"predicted_collision_{step*dt:.1f}s", obj)
                break

    def get_current_status(self) -> dict:
        end_effector_alert = AlertLevel.SAFE
        target_alert = AlertLevel.SAFE
        
        if self.current_end_effector:
            end_effector_alert, _ = self.zone_manager.check_all_zones(self.current_end_effector.position)
        
        if self.current_target:
            target_alert, _ = self.zone_manager.check_all_zones(self.current_target.position)
        
        return {
            "end_effector": {
                "position": self.current_end_effector.position if self.current_end_effector else None,
                "alert_level": end_effector_alert.value
            },
            "target": {
                "position": self.current_target.position if self.current_target else None,
                "alert_level": target_alert.value
            },
            "active_zones": [z.name for z in self.zone_manager.list_zones() if z.enabled]
        }


class SafeArmController:
    def __init__(self, mc, zone_manager: ZoneManager, safety_monitor: SafetyMonitor):
        self.mc = mc
        self.zone_manager = zone_manager
        self.safety_monitor = safety_monitor
        self.path_planner = PathPlanner(zone_manager)
        self.emergency_stop_triggered = False
        self._last_coords: Optional[List[float]] = None

    def _update_position_from_arm(self) -> None:
        try:
            coords = self.mc.get_coords()
            if coords and len(coords) >= 3:
                self._last_coords = coords
                self.safety_monitor.update_end_effector(
                    (coords[0], coords[1], coords[2])
                )
        except Exception as e:
            print(f"获取机械臂位置失败: {e}")

    def send_coords(self, coords: List[float], speed: int, mode: int = 0) -> int:
        if self.emergency_stop_triggered:
            print("紧急停止已触发，禁止移动")
            return 0
        
        if len(coords) < 3:
            print("坐标格式错误")
            return 0
        
        target_pos = (coords[0], coords[1], coords[2])
        
        self._update_position_from_arm()
        
        if self._last_coords and len(self._last_coords) >= 3:
            start_pos = (self._last_coords[0], self._last_coords[1], self._last_coords[2])
        else:
            start_pos = (0, 0, self.path_planner.safe_height)
        
        target_alert, zone_alerts = self.zone_manager.check_all_zones(target_pos)
        
        if target_alert in [AlertLevel.DANGER, AlertLevel.EMERGENCY]:
            dangerous_zones = [z.name for z, l in zone_alerts if l in [AlertLevel.DANGER, AlertLevel.EMERGENCY]]
            print(f"目标位置在危险区域内: {dangerous_zones}")
            return 0
        
        path_safe, _ = self.path_planner.check_path_safe(start_pos, target_pos)
        
        if not path_safe:
            print("检测到路径风险，规划绕行路径...")
            waypoints = self.path_planner.plan_alternate_path(start_pos, target_pos, current_z=start_pos[2])
            
            for i, waypoint in enumerate(waypoints[:-1]):
                next_waypoint = waypoints[i + 1]
                
                segment_safe, _ = self.path_planner.check_path_safe(waypoint, next_waypoint)
                
                if not segment_safe:
                    print(f"无法规划安全路径，段 {i+1} 存在风险")
                    return 0
                
                full_waypoint = list(waypoint) + coords[3:] if len(coords) > 3 else list(waypoint)
                result = self.mc.send_coords(full_waypoint, speed, mode)
                
                current_pos = (next_waypoint[0], next_waypoint[1], next_waypoint[2])
                self.safety_monitor.update_end_effector(current_pos)
                self._last_coords = full_waypoint
                
                time.sleep(2)
            
            return 1
        
        result = self.mc.send_coords(coords, speed, mode)
        self.safety_monitor.update_end_effector(target_pos)
        self._last_coords = coords
        
        return result

    def safe_send_coords(self, coords: List[float], speed: int = 20, mode: int = 0,
                          height_safe: float = 220.0) -> Tuple[bool, str]:
        result = self.send_coords(coords, speed, mode)
        if result == 1:
            return True, "移动成功"
        else:
            return False, "移动被阻止或失败"

    def get_coords(self) -> List[float]:
        coords = self.mc.get_coords()
        if coords and len(coords) >= 3:
            self._last_coords = coords
            self.safety_monitor.update_end_effector((coords[0], coords[1], coords[2]))
        return coords

    def send_angles(self, angles: List[float], speed: int) -> int:
        if self.emergency_stop_triggered:
            print("紧急停止已触发，禁止移动")
            return 0
        
        result = self.mc.send_angles(angles, speed)
        time.sleep(0.5)
        self._update_position_from_arm()
        return result

    def send_angle(self, joint_id: int, angle: float, speed: int) -> int:
        if self.emergency_stop_triggered:
            print("紧急停止已触发，禁止移动")
            return 0
        
        result = self.mc.send_angle(joint_id, angle, speed)
        time.sleep(0.5)
        self._update_position_from_arm()
        return result

    def get_angles(self) -> List[float]:
        return self.mc.get_angles()

    def send_coord(self, coord_id: int, coord: float, speed: int) -> int:
        if self.emergency_stop_triggered:
            print("紧急停止已触发，禁止移动")
            return 0
        
        result = self.mc.send_coord(coord_id, coord, speed)
        time.sleep(0.5)
        self._update_position_from_arm()
        return result

    def get_system_version(self) -> float:
        return self.mc.get_system_version()

    def get_atom_version(self) -> float:
        return self.mc.get_atom_version()

    def power_on(self) -> int:
        return self.mc.power_on()

    def power_off(self) -> int:
        return self.mc.power_off()

    def is_power_on(self) -> int:
        return self.mc.is_power_on()

    def release_all_servos(self, data: int = None) -> int:
        if data is not None:
            return self.mc.release_all_servos(data)
        return self.mc.release_all_servos()

    def focus_servo(self, servo_id: int) -> int:
        return self.mc.focus_servo(servo_id)

    def focus_all_servos(self) -> int:
        return self.mc.focus_all_servos()

    def get_fresh_mode(self) -> int:
        return self.mc.get_fresh_mode()

    def set_fresh_mode(self, mode: int) -> int:
        return self.mc.set_fresh_mode(mode)

    def get_robot_status(self) -> List[int]:
        return self.mc.get_robot_status()

    def read_next_error(self) -> List[int]:
        return self.mc.read_next_error()

    def is_controller_connected(self) -> int:
        return self.mc.is_controller_connected()

    def safe_pump_move(self, XY_START: List[float], XY_END: List[float],
                        HEIGHT_START: float = 90, HEIGHT_END: float = 100,
                        HEIGHT_SAFE: float = 220) -> Tuple[bool, str]:
        if self.emergency_stop_triggered:
            return False, "紧急停止已触发，禁止移动"
        
        start_xy = (XY_START[0], XY_START[1], HEIGHT_SAFE)
        end_xy = (XY_END[0], XY_END[1], HEIGHT_SAFE)
        
        for pos, name in [(start_xy, "起点上方"), (end_xy, "终点上方")]:
            alert, _ = self.zone_manager.check_all_zones(pos)
            if alert in [AlertLevel.DANGER, AlertLevel.EMERGENCY]:
                return False, f"{name}位置处于危险区域"
        
        print("安全检查通过，执行吸泵移动...")
        
        try:
            from utils_pump import pump_on, pump_off
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(20, GPIO.OUT)
            GPIO.setup(21, GPIO.OUT)
        except ImportError:
            pass
        
        success, msg = self.safe_send_coords(
            [XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90],
            speed=20
        )
        if not success:
            return False, f"移动到起点上方失败: {msg}"
        time.sleep(4)
        
        try:
            pump_on()
        except:
            pass
        
        success, msg = self.safe_send_coords(
            [XY_START[0], XY_START[1], HEIGHT_START, 0, 180, 90],
            speed=15
        )
        if not success:
            return False, f"向下吸取失败: {msg}"
        time.sleep(4)
        
        success, msg = self.safe_send_coords(
            [XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90],
            speed=15
        )
        if not success:
            return False, f"升起物体失败: {msg}"
        time.sleep(4)
        
        success, msg = self.safe_send_coords(
            [XY_END[0], XY_END[1], HEIGHT_SAFE, 0, 180, 90],
            speed=15
        )
        if not success:
            return False, f"移动到终点上方失败: {msg}"
        time.sleep(4)
        
        success, msg = self.safe_send_coords(
            [XY_END[0], XY_END[1], HEIGHT_END, 0, 180, 90],
            speed=20
        )
        if not success:
            return False, f"向下放下物体失败: {msg}"
        time.sleep(3)
        
        try:
            pump_off()
        except:
            pass
        
        return True, "安全吸泵移动完成"

    def emergency_stop(self) -> None:
        self.emergency_stop_triggered = True
        try:
            self.mc.release_all_servos()
        except:
            pass
        print("紧急停止已触发")

    def reset_emergency(self) -> None:
        self.emergency_stop_triggered = False
        print("紧急停止已重置")


@dataclass
class VLMDetector:
    config: VLMConfig = field(default_factory=VLMConfig)
    _client: Optional[Any] = None
    _last_call_time: float = 0.0
    min_call_interval: float = 2.0

    def __post_init__(self):
        if OPENAI_AVAILABLE and self.config.api_key:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url if self.config.base_url else None
            )

    def is_available(self) -> bool:
        return self._client is not None and OPENAI_AVAILABLE

    def _frame_to_base64(self, frame) -> str:
        if CV2_AVAILABLE:
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        return ""

    def detect_objects(self, frame, custom_prompt: str = None) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            return None
        
        current_time = time.time()
        if current_time - self._last_call_time < self.min_call_interval:
            return None
        
        self._last_call_time = current_time
        
        try:
            base64_image = self._frame_to_base64(frame)
            prompt = custom_prompt if custom_prompt else VLM_PROMPT_DETECT_OBJECTS
            
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            content = response.choices[0].message.content.strip()
            
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                return result
            
            return None
            
        except Exception as e:
            print(f"VLM 检测错误: {e}")
            return None


class RealTimeVisualAnalyzer:
    def __init__(self, safety_monitor: SafetyMonitor, zone_manager: ZoneManager, eye2hand_func):
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 未安装，无法使用 RealTimeVisualAnalyzer")
        
        self.safety_monitor = safety_monitor
        self.zone_manager = zone_manager
        self.eye2hand = eye2hand_func
        self._running = False
        self._analyze_thread: Optional[threading.Thread] = None
        self.last_frame = None
        self.detection_results = None
        self.vlm_detector: Optional[VLMDetector] = None
        self._last_detection_time: float = 0.0
        self.detection_interval: float = 3.0
        self.height_estimate: float = 100.0
        self._prev_end_effector_pos: Optional[Tuple[float, float, float]] = None
        self._prev_target_pos: Optional[Tuple[float, float, float]] = None

    def configure_vlm(self, config: VLMConfig) -> None:
        self.vlm_detector = VLMDetector(config=config)
        print("VLM 检测器已配置")

    def start_analysis(self, camera_source: int = 0, analyze_interval: float = 0.1) -> None:
        if self._running:
            return
        
        self._running = True
        self._analyze_thread = threading.Thread(
            target=self._analysis_loop,
            args=(camera_source, analyze_interval),
            daemon=True
        )
        self._analyze_thread.start()

    def stop_analysis(self) -> None:
        self._running = False
        if self._analyze_thread:
            self._analyze_thread.join(timeout=2.0)
            self._analyze_thread = None

    def _analysis_loop(self, camera_source: int, interval: float) -> None:
        cap = cv2.VideoCapture(camera_source)
        cap.open(camera_source)
        
        while self._running:
            try:
                success, img_bgr = cap.read()
                if success:
                    self.last_frame = img_bgr.copy()
                    self._analyze_frame(img_bgr)
            except Exception as e:
                print(f"视觉分析错误: {e}")
            
            time.sleep(interval)
        
        cap.release()

    def _analyze_frame(self, frame) -> None:
        current_time = time.time()
        
        if self.vlm_detector and self.vlm_detector.is_available():
            if current_time - self._last_detection_time >= self.detection_interval:
                self._last_detection_time = current_time
                
                result = self.vlm_detector.detect_objects(frame)
                if result:
                    self.detection_results = result
                    self._process_detection_result(result, frame)

    def _process_detection_result(self, result: Dict[str, Any], frame) -> None:
        if "objects" not in result:
            return
        
        img_h, img_w = frame.shape[:2]
        
        for obj in result.get("objects", []):
            name = obj.get("name", "")
            bbox_xyxy = obj.get("bbox_xyxy", [[0, 0], [0, 0]])
            confidence = obj.get("confidence", 0.0)
            
            if confidence < 0.5:
                continue
            
            self.update_object_from_vlm(name, bbox_xyxy, img_w, img_h, confidence)

    def update_object_from_vlm(self, object_name: str, bbox_xyxy: List[List[float]], 
                                img_w: int, img_h: int, confidence: float = 1.0,
                                height_estimate: float = None) -> None:
        if height_estimate is None:
            height_estimate = self.height_estimate
        
        x_min = int(bbox_xyxy[0][0] * img_w / 999)
        y_min = int(bbox_xyxy[0][1] * img_h / 999)
        x_max = int(bbox_xyxy[1][0] * img_w / 999)
        y_max = int(bbox_xyxy[1][1] * img_h / 999)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        mc_x, mc_y = self.eye2hand(x_center, y_center)
        current_pos = (mc_x, mc_y, height_estimate)
        
        velocity = (0.0, 0.0, 0.0)
        current_time = time.time()
        
        if "end_effector" in object_name.lower() or "机械臂" in object_name:
            if self._prev_end_effector_pos:
                dt = current_time - self._last_detection_time
                if dt > 0:
                    velocity = (
                        (current_pos[0] - self._prev_end_effector_pos[0]) / dt,
                        (current_pos[1] - self._prev_end_effector_pos[1]) / dt,
                        (current_pos[2] - self._prev_end_effector_pos[2]) / dt
                    )
            
            self.safety_monitor.update_end_effector(current_pos, velocity, confidence)
            self._prev_end_effector_pos = current_pos
        else:
            if self._prev_target_pos:
                dt = current_time - self._last_detection_time
                if dt > 0:
                    velocity = (
                        (current_pos[0] - self._prev_target_pos[0]) / dt,
                        (current_pos[1] - self._prev_target_pos[1]) / dt,
                        (current_pos[2] - self._prev_target_pos[2]) / dt
                    )
            
            self.safety_monitor.update_target(current_pos, velocity, confidence)
            self._prev_target_pos = current_pos

    def get_visualization_overlay(self, frame) -> np.ndarray:
        overlay = frame.copy()
        
        for zone in self.zone_manager.list_zones():
            if not zone.enabled:
                continue
            
            color = zone.color
            
            if zone.zone_type == ZoneType.SPHERE:
                center_2d = self._project_3d_to_2d(zone.center)
                if center_2d:
                    radius_2d = int(zone.dimensions[0] * 0.5)
                    cv2.circle(overlay, center_2d, radius_2d, color, 2)
                    cv2.putText(overlay, zone.name, (center_2d[0] - 30, center_2d[1] - radius_2d - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            elif zone.zone_type == ZoneType.RECTANGLE:
                top_left_2d = self._project_3d_to_2d((
                    zone.center[0] - zone.dimensions[0]/2,
                    zone.center[1] - zone.dimensions[1]/2,
                    zone.center[2]
                ))
                bottom_right_2d = self._project_3d_to_2d((
                    zone.center[0] + zone.dimensions[0]/2,
                    zone.center[1] + zone.dimensions[1]/2,
                    zone.center[2]
                ))
                if top_left_2d and bottom_right_2d:
                    cv2.rectangle(overlay, top_left_2d, bottom_right_2d, color, 2)
            
            elif zone.zone_type == ZoneType.CYLINDER:
                center_2d = self._project_3d_to_2d(zone.center)
                if center_2d:
                    radius_2d = int(zone.dimensions[0] * 0.5)
                    cv2.circle(overlay, center_2d, radius_2d, color, 2)
                    cv2.putText(overlay, zone.name, (center_2d[0] - 30, center_2d[1] - radius_2d - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if self.detection_results and "objects" in self.detection_results:
            img_h, img_w = frame.shape[:2]
            
            for obj in self.detection_results.get("objects", []):
                name = obj.get("name", "")
                bbox_xyxy = obj.get("bbox_xyxy", [[0, 0], [0, 0]])
                confidence = obj.get("confidence", 0.0)
                
                x_min = int(bbox_xyxy[0][0] * img_w / 999)
                y_min = int(bbox_xyxy[0][1] * img_h / 999)
                x_max = int(bbox_xyxy[1][0] * img_w / 999)
                y_max = int(bbox_xyxy[1][1] * img_h / 999)
                
                if "end_effector" in name.lower():
                    box_color = (0, 255, 0)
                else:
                    box_color = (255, 165, 0)
                
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(overlay, f"{name} ({confidence:.2f})", 
                           (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        return overlay

    def _project_3d_to_2d(self, point_3d: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
        try:
            cali_1_mc = [-21.8, -197.4]
            cali_1_im = [130, 290]
            cali_2_mc = [215, -59.1]
            cali_2_im = [640, 0]
            
            x_mc, y_mc, z_mc = point_3d
            
            x_im = int(np.interp(x_mc, [cali_1_mc[0], cali_2_mc[0]], [cali_1_im[0], cali_2_im[0]]))
            y_im = int(np.interp(y_mc, [cali_2_mc[1], cali_1_mc[1]], [cali_2_im[1], cali_1_im[1]]))
            
            return (x_im, y_im)
        except:
            return None


def create_default_zone_manager() -> ZoneManager:
    zone_manager = ZoneManager()
    
    table_corner = SafetyZone(
        name="桌面角落禁区",
        zone_type=ZoneType.RECTANGLE,
        center=(-100, -150, 100),
        dimensions=(100, 100, 200),
        warning_distance=30,
        danger_distance=15,
        color=(0, 0, 255)
    )
    zone_manager.add_zone(table_corner)
    
    central_pillar = SafetyZone(
        name="中心支柱禁区",
        zone_type=ZoneType.CYLINDER,
        center=(50, -100, 100),
        dimensions=(50, 200, 0),
        warning_distance=25,
        danger_distance=10,
        color=(255, 0, 0)
    )
    zone_manager.add_zone(central_pillar)
    
    return zone_manager


def alert_callback_example(level: AlertLevel, zone_name: str, obj: ObjectState) -> None:
    level_names = {
        AlertLevel.SAFE: "安全",
        AlertLevel.WARNING: "警告",
        AlertLevel.DANGER: "危险",
        AlertLevel.EMERGENCY: "紧急"
    }
    
    print(f"[{level_names[level]}] 对象 {obj.name} 接近区域 {zone_name}")
    print(f"    当前位置: {obj.position}")
    
    if level == AlertLevel.DANGER:
        print("    ⚠️  警告：请调整路径")
    elif level == AlertLevel.EMERGENCY:
        print("    🚨 紧急：已进入禁区！")
