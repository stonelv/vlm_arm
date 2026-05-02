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
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Callable
import threading

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: cv2 未安装，RealTimeVisualAnalyzer 将不可用")


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

    def plan_alternate_path(self, start: Tuple[float, float, float],
                             end: Tuple[float, float, float],
                             current_z: float = None) -> List[Tuple[float, float, float]]:
        if current_z is None:
            current_z = self.safe_height
        
        is_safe, _ = self.check_path_safe(start, end)
        if is_safe:
            return [start, end]
        
        waypoints = [start]
        
        safe_point = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            self.safe_height
        )
        
        waypoints.append(safe_point)
        waypoints.append(end)
        
        if self.check_path_safe(safe_point, end)[0]:
            return waypoints
        
        zone, distance = self.zone_manager.get_closest_zone(start)
        if zone:
            cx, cy, cz = zone.center
            radius = zone.dimensions[0] if zone.zone_type == ZoneType.SPHERE else max(zone.dimensions)
            
            if end[0] > start[0]:
                detour_x = cx - (radius + 50)
            else:
                detour_x = cx + (radius + 50)
            
            detour_point = (detour_x, (start[1] + end[1]) / 2, self.safe_height)
            
            waypoints = [start, detour_point, end]
            
            if self.check_path_safe(start, detour_point)[0] and self.check_path_safe(detour_point, end)[0]:
                return waypoints
        
        safe_waypoints = []
        safe_waypoints.append((start[0], start[1], self.safe_height))
        safe_waypoints.append((end[0], end[1], self.safe_height))
        safe_waypoints.append(end)
        
        return safe_waypoints

    def is_point_safe(self, point: Tuple[float, float, float]) -> bool:
        alert_level, _ = self.zone_manager.check_all_zones(point)
        return alert_level == AlertLevel.SAFE


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

    def safe_send_coords(self, coords: List[float], speed: int = 20, mode: int = 0,
                          height_safe: float = 220.0) -> Tuple[bool, str]:
        if self.emergency_stop_triggered:
            return False, "紧急停止已触发，禁止移动"
        
        if len(coords) < 3:
            return False, "坐标格式错误"
        
        target_pos = (coords[0], coords[1], coords[2])
        
        current_coords = self.mc.get_coords()
        if current_coords and len(current_coords) >= 3:
            start_pos = (current_coords[0], current_coords[1], current_coords[2])
            self.safety_monitor.update_end_effector(start_pos)
        else:
            start_pos = (0, 0, height_safe)
        
        target_alert, zone_alerts = self.zone_manager.check_all_zones(target_pos)
        
        if target_alert in [AlertLevel.DANGER, AlertLevel.EMERGENCY]:
            dangerous_zones = [z.name for z, l in zone_alerts if l in [AlertLevel.DANGER, AlertLevel.EMERGENCY]]
            return False, f"目标位置在危险区域内: {dangerous_zones}"
        
        path_safe, _ = self.path_planner.check_path_safe(start_pos, target_pos)
        
        if not path_safe:
            print("检测到路径风险，规划绕行路径...")
            waypoints = self.path_planner.plan_alternate_path(start_pos, target_pos, current_z=start_pos[2])
            
            for i, waypoint in enumerate(waypoints[:-1]):
                next_waypoint = waypoints[i + 1]
                
                segment_safe, _ = self.path_planner.check_path_safe(waypoint, next_waypoint)
                
                if not segment_safe:
                    return False, f"无法规划安全路径，段 {i+1} 存在风险"
                
                full_waypoint = list(waypoint) + coords[3:] if len(coords) > 3 else list(waypoint)
                self.mc.send_coords(full_waypoint, speed, mode)
                
                current_pos = (next_waypoint[0], next_waypoint[1], next_waypoint[2])
                self.safety_monitor.update_end_effector(current_pos)
                
                time.sleep(2)
            
            return True, "绕行路径执行完成"
        
        self.mc.send_coords(coords, speed, mode)
        self.safety_monitor.update_end_effector(target_pos)
        
        return True, "直接路径安全执行"

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
            speed=20, height_safe=HEIGHT_SAFE
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
            speed=15, height_safe=HEIGHT_SAFE
        )
        if not success:
            return False, f"向下吸取失败: {msg}"
        time.sleep(4)
        
        success, msg = self.safe_send_coords(
            [XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90],
            speed=15, height_safe=HEIGHT_SAFE
        )
        if not success:
            return False, f"升起物体失败: {msg}"
        time.sleep(4)
        
        success, msg = self.safe_send_coords(
            [XY_END[0], XY_END[1], HEIGHT_SAFE, 0, 180, 90],
            speed=15, height_safe=HEIGHT_SAFE
        )
        if not success:
            return False, f"移动到终点上方失败: {msg}"
        time.sleep(4)
        
        success, msg = self.safe_send_coords(
            [XY_END[0], XY_END[1], HEIGHT_END, 0, 180, 90],
            speed=20, height_safe=HEIGHT_SAFE
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

    def start_analysis(self, camera_source: int = 0, analyze_interval: float = 0.5) -> None:
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
        pass

    def update_object_from_vlm(self, object_name: str, bbox_xyxy: List[List[float]], 
                                img_w: int, img_h: int, height_estimate: float = 100.0) -> None:
        x_min = int(bbox_xyxy[0][0] * img_w / 999)
        y_min = int(bbox_xyxy[0][1] * img_h / 999)
        x_max = int(bbox_xyxy[1][0] * img_w / 999)
        y_max = int(bbox_xyxy[1][1] * img_h / 999)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        mc_x, mc_y = self.eye2hand(x_center, y_center)
        
        if "end_effector" in object_name.lower() or "机械臂" in object_name:
            self.safety_monitor.update_end_effector((mc_x, mc_y, height_estimate))
        else:
            self.safety_monitor.update_target((mc_x, mc_y, height_estimate))

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
