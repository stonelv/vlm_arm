# utils_tracking.py
# 实时目标追踪行为模块
# 功能：驱动机械臂根据摄像头实时画面追踪移动的人脸或指定物品

print('导入实时目标追踪模块')

import cv2
import numpy as np
import time
import threading
from collections import deque

from utils_robot import *
from utils_asr import *
from utils_tts import *
from utils_vlm import *

class TrackingConfig:
    TRACKING_TARGET_TYPES = {
        'face': {'name': '人脸', 'priority': 100},
        'red_box': {'name': '红色盒子', 'priority': 90, 'color_range': ([0, 100, 100], [10, 255, 255]), 'color_range2': ([160, 100, 100], [180, 255, 255])},
        'green_box': {'name': '绿色盒子', 'priority': 80, 'color_range': ([35, 50, 50], [85, 255, 255])},
        'blue_box': {'name': '蓝色盒子', 'priority': 70, 'color_range': ([100, 50, 50], [130, 255, 255])},
        'yellow_box': {'name': '黄色盒子', 'priority': 60, 'color_range': ([20, 100, 100], [30, 255, 255])},
    }
    
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    DESIRED_DISTANCE = 150
    DESIRED_HEIGHT = 120
    DISTANCE_TOLERANCE = 10
    HEIGHT_TOLERANCE = 10
    POSITION_TOLERANCE = 20
    
    MOVEMENT_SMOOTHING = 0.3
    MAX_SPEED_XY = 20
    MAX_SPEED_Z = 10
    
    OCCLUSION_TIMEOUT = 2.0
    MAX_OCCLUSION_PREDICTION_STEPS = 30
    
    PITCH_ANGLE = -45
    GRIPPER_ROTATION = 0

class TargetState:
    def __init__(self):
        self.is_tracking = False
        self.current_target_type = None
        self.current_target_bbox = None
        self.current_target_center = None
        self.previous_positions = deque(maxlen=10)
        self.velocity = np.array([0, 0, 0])
        self.is_occluded = False
        self.occlusion_start_time = 0
        self.predicted_position = None

class VisualDetector:
    def __init__(self):
        self.face_cascade = self._load_face_detector()
        self.tracking_config = TrackingConfig()
        
    def _load_face_detector(self):
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            print('警告：人脸检测器加载失败')
        return face_cascade
    
    def detect_face(self, frame):
        if self.face_cascade.empty():
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            return {
                'bbox': [x, y, x + w, y + h],
                'center': (x + w // 2, y + h // 2),
                'area': w * h,
                'type': 'face'
            }
        return None
    
    def detect_color_object(self, frame, target_type):
        config = self.tracking_config.TRACKING_TARGET_TYPES.get(target_type)
        if not config or 'color_range' not in config:
            return None
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_range = config['color_range']
        lower = np.array(color_range[0])
        upper = np.array(color_range[1])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        if 'color_range2' in config:
            color_range2 = config['color_range2']
            lower2 = np.array(color_range2[0])
            upper2 = np.array(color_range2[1])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            c = contours[0]
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            if area > 500:
                return {
                    'bbox': [x, y, x + w, y + h],
                    'center': (x + w // 2, y + h // 2),
                    'area': area,
                    'type': target_type,
                    'mask': mask
                }
        return None
    
    def detect_all_targets(self, frame):
        detected_targets = []
        
        face_result = self.detect_face(frame)
        if face_result:
            detected_targets.append(face_result)
        
        for target_type in self.tracking_config.TRACKING_TARGET_TYPES:
            if target_type == 'face':
                continue
            
            result = self.detect_color_object(frame, target_type)
            if result:
                detected_targets.append(result)
        
        detected_targets.sort(
            key=lambda x: self.tracking_config.TRACKING_TARGET_TYPES[x['type']]['priority'],
            reverse=True
        )
        
        return detected_targets

class MotionController:
    def __init__(self, mc):
        self.mc = mc
        self.config = TrackingConfig()
        self.current_coords = None
        self.target_coords = None
        self.is_initialized = False
        
    def initialize_position(self):
        print('初始化追踪姿态')
        self.mc.send_angles([0, -30, 60, 0, 90, 0], 30)
        time.sleep(2)
        
        coords = self.mc.get_coords()
        if coords and len(coords) >= 3:
            self.current_coords = np.array(coords[:3], dtype=np.float64)
        else:
            self.current_coords = np.array([200, 0, 150], dtype=np.float64)
        
        self.is_initialized = True
        print(f'当前初始坐标: {self.current_coords}')
    
    def pixel_to_world(self, pixel_center, frame, detected_target):
        img_center_x = self.config.CAMERA_WIDTH // 2
        img_center_y = self.config.CAMERA_HEIGHT // 2
        
        offset_x = pixel_center[0] - img_center_x
        offset_y = pixel_center[1] - img_center_y
        
        scale_factor = 0.2
        world_offset_x = offset_x * scale_factor
        world_offset_y = offset_y * scale_factor
        
        target_area = detected_target.get('area', 10000)
        distance_estimate = max(100, min(300, 25000 / np.sqrt(target_area)))
        
        desired_x = self.current_coords[0] - world_offset_y
        desired_y = self.current_coords[1] + world_offset_x
        
        height_adjustment = 0
        if offset_y < -30:
            height_adjustment = 5
        elif offset_y > 30:
            height_adjustment = -5
        
        desired_z = self.current_coords[2] + height_adjustment
        
        return np.array([desired_x, desired_y, desired_z])
    
    def smooth_movement(self, current, target, smoothing_factor):
        return current + (target - current) * smoothing_factor
    
    def move_to_target(self, target_world_coords, frame, detected_target):
        if not self.is_initialized:
            self.initialize_position()
        
        smoothed_target = self.smooth_movement(
            self.current_coords, 
            target_world_coords, 
            self.config.MOVEMENT_SMOOTHING
        )
        
        delta = smoothed_target - self.current_coords
        distance = np.linalg.norm(delta)
        
        if distance > self.config.POSITION_TOLERANCE:
            max_speed = self.config.MAX_SPEED_XY
            if distance > max_speed:
                delta = (delta / distance) * max_speed
            
            new_coords = self.current_coords + delta
            
            new_coords[0] = np.clip(new_coords[0], 100, 350)
            new_coords[1] = np.clip(new_coords[1], -200, 200)
            new_coords[2] = np.clip(new_coords[2], 50, 250)
            
            try:
                current_pose = self.mc.get_coords()
                if current_pose and len(current_pose) >= 6:
                    rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]
                else:
                    rx, ry, rz = 0, 180, 90
                
                self.mc.send_coords(
                    [new_coords[0], new_coords[1], new_coords[2], rx, ry, rz],
                    20, 0
                )
                
                self.current_coords = new_coords
                
            except Exception as e:
                print(f'运动控制错误: {e}')
            
            return True
        return False
    
    def maintain_position(self):
        pass
    
    def stop_movement(self):
        print('停止机械臂运动')
        self.mc.release_all_servos()
        time.sleep(0.5)

class OcclusionHandler:
    def __init__(self, target_state, config):
        self.target_state = target_state
        self.config = config
        self.prediction_model = None
        
    def handle_occlusion(self):
        if not self.target_state.is_occluded:
            self.target_state.is_occluded = True
            self.target_state.occlusion_start_time = time.time()
            print('目标被遮挡，开始预测运动')
        
        occlusion_duration = time.time() - self.target_state.occlusion_start_time
        
        if occlusion_duration > self.config.OCCLUSION_TIMEOUT:
            print('遮挡超时，丢失目标')
            return False
        
        self._predict_position()
        return True
    
    def _predict_position(self):
        if len(self.target_state.previous_positions) < 3:
            return
        
        positions = np.array(list(self.target_state.previous_positions))
        recent_positions = positions[-5:]
        
        if len(recent_positions) >= 2:
            velocities = np.diff(recent_positions, axis=0)
            avg_velocity = np.mean(velocities, axis=0)
            
            prediction_steps = min(
                int((time.time() - self.target_state.occlusion_start_time) / 0.1),
                self.config.MAX_OCCLUSION_PREDICTION_STEPS
            )
            
            last_position = recent_positions[-1]
            predicted = last_position + avg_velocity * prediction_steps
            
            self.target_state.predicted_position = predicted
            print(f'预测目标位置: {predicted}')
    
    def recover_tracking(self, detected_target):
        if self.target_state.is_occluded:
            print('目标重新出现，恢复追踪')
            self.target_state.is_occluded = False
            self.target_state.predicted_position = None

class MultiTargetManager:
    def __init__(self, target_state, config):
        self.target_state = target_state
        self.config = config
        self.manual_override = False
        self.manual_target = None
        
    def select_target(self, detected_targets):
        if self.manual_override and self.manual_target:
            for target in detected_targets:
                if target['type'] == self.manual_target:
                    return target
            return None
        
        if len(detected_targets) > 0:
            return detected_targets[0]
        return None
    
    def switch_target(self, target_type):
        print(f'手动切换追踪目标为: {target_type}')
        self.manual_override = True
        self.manual_target = target_type
        self.target_state.current_target_type = target_type
        
    def release_override(self):
        self.manual_override = False
        self.manual_target = None
        print('恢复自动目标选择')

class VoiceCommandHandler:
    def __init__(self, multi_target_manager, target_state):
        self.multi_target_manager = multi_target_manager
        self.target_state = target_state
        self.command_keywords = {
            '红色': 'red_box',
            '红色盒子': 'red_box',
            '红盒子': 'red_box',
            '绿色': 'green_box',
            '绿色盒子': 'green_box',
            '绿盒子': 'green_box',
            '蓝色': 'blue_box',
            '蓝色盒子': 'blue_box',
            '蓝盒子': 'blue_box',
            '黄色': 'yellow_box',
            '黄色盒子': 'yellow_box',
            '黄盒子': 'yellow_box',
            '人脸': 'face',
            '脸': 'face',
            '人': 'face',
        }
        
        self.action_keywords = {
            '跟着': 'follow',
            '追踪': 'follow',
            '跟踪': 'follow',
            '看': 'follow',
            '开始': 'start',
            '停止': 'stop',
            '结束': 'stop',
            '暂停': 'pause',
            '自动': 'auto',
            '恢复': 'auto',
        }
    
    def listen_and_process(self):
        try:
            print('正在监听语音指令...')
            record_auto()
            command = speech_recognition()
            
            if command:
                return self.process_command(command)
        except Exception as e:
            print(f'语音处理错误: {e}')
        
        return None
    
    def process_command(self, command):
        command_lower = command.lower()
        print(f'处理语音指令: {command}')
        
        action = None
        target = None
        
        for keyword, act in self.action_keywords.items():
            if keyword in command_lower:
                action = act
                break
        
        for keyword, tgt in self.command_keywords.items():
            if keyword in command_lower:
                target = tgt
                break
        
        return {
            'action': action,
            'target': target,
            'original_command': command
        }
    
    def execute_command(self, command_result):
        if not command_result:
            return False
        
        action = command_result.get('action')
        target = command_result.get('target')
        original = command_result.get('original_command')
        
        print(f'执行命令: action={action}, target={target}')
        
        if action == 'stop':
            tts('好的，停止追踪')
            play_wav('temp/tts.wav')
            self.target_state.is_tracking = False
            return True
            
        elif action == 'start':
            tts('好的，开始追踪')
            play_wav('temp/tts.wav')
            self.target_state.is_tracking = True
            return True
            
        elif action == 'auto':
            tts('好的，切换到自动模式')
            play_wav('temp/tts.wav')
            self.multi_target_manager.release_override()
            return True
            
        elif action == 'follow' and target:
            target_name = TrackingConfig.TRACKING_TARGET_TYPES[target]['name']
            tts(f'好的，现在开始追踪{target_name}')
            play_wav('temp/tts.wav')
            self.multi_target_manager.switch_target(target)
            self.target_state.is_tracking = True
            return True
        
        return False

class TrackingVisualizer:
    def __init__(self, config):
        self.config = config
        
    def draw_tracking_overlay(self, frame, target_state, detected_targets, selected_target):
        output_frame = frame.copy()
        
        for target in detected_targets:
            bbox = target['bbox']
            target_type = target['type']
            priority = self.config.TRACKING_TARGET_TYPES[target_type]['priority']
            name = self.config.TRACKING_TARGET_TYPES[target_type]['name']
            
            color = (0, 255, 0) if target == selected_target else (128, 128, 128)
            thickness = 3 if target == selected_target else 1
            
            cv2.rectangle(output_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            label = f'{name} (P:{priority})'
            cv2.putText(output_frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if target_state.is_occluded and target_state.predicted_position is not None:
            pred_center = (int(target_state.predicted_position[0]), 
                          int(target_state.predicted_position[1]))
            cv2.circle(output_frame, pred_center, 20, (0, 165, 255), 2)
            cv2.putText(output_frame, '预测位置', (pred_center[0] - 30, pred_center[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        status_text = '追踪中' if target_state.is_tracking else '等待中'
        status_color = (0, 255, 0) if target_state.is_tracking else (0, 0, 255)
        
        if target_state.is_occluded:
            status_text = '遮挡中'
            status_color = (0, 165, 255)
            
        cv2.putText(output_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if target_state.current_target_type:
            target_name = self.config.TRACKING_TARGET_TYPES[target_state.current_target_type]['name']
            cv2.putText(output_frame, f'目标: {target_name}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_frame

class RealTimeTracker:
    def __init__(self):
        self.config = TrackingConfig()
        self.target_state = TargetState()
        
        self.visual_detector = VisualDetector()
        self.motion_controller = MotionController(mc)
        self.occlusion_handler = OcclusionHandler(self.target_state, self.config)
        self.multi_target_manager = MultiTargetManager(self.target_state, self.config)
        self.voice_handler = VoiceCommandHandler(self.multi_target_manager, self.target_state)
        self.visualizer = TrackingVisualizer(self.config)
        
        self.is_running = False
        self.voice_thread = None
        self.cap = None
        
    def initialize(self):
        print('初始化实时追踪系统...')
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        if not self.cap.isOpened():
            raise RuntimeError('无法打开摄像头')
        
        self.motion_controller.initialize_position()
        
        print('实时追踪系统初始化完成')
        tts('实时追踪系统已准备就绪')
        play_wav('temp/tts.wav')
        
    def _voice_listener_loop(self):
        while self.is_running:
            try:
                command_result = self.voice_handler.listen_and_process()
                if command_result:
                    self.voice_handler.execute_command(command_result)
            except Exception as e:
                print(f'语音监听错误: {e}')
                time.sleep(1)
    
    def start(self):
        self.is_running = True
        self.target_state.is_tracking = True
        
        self.voice_thread = threading.Thread(target=self._voice_listener_loop, daemon=True)
        self.voice_thread.start()
        
        print('开始实时追踪...')
        print('控制说明:')
        print('  - 按 q 键退出')
        print('  - 按 空格键 开始/暂停追踪')
        print('  - 语音指令: "跟着红色盒子", "停止追踪", "恢复自动"')
        
        frame_count = 0
        last_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print('无法获取摄像头画面')
                break
            
            frame_count += 1
            if time.time() - last_time > 1:
                fps = frame_count / (time.time() - last_time)
                print(f'FPS: {fps:.1f}')
                frame_count = 0
                last_time = time.time()
            
            detected_targets = self.visual_detector.detect_all_targets(frame)
            
            if self.target_state.is_tracking:
                selected_target = self.multi_target_manager.select_target(detected_targets)
                
                if selected_target:
                    self.occlusion_handler.recover_tracking(selected_target)
                    self.target_state.current_target_type = selected_target['type']
                    self.target_state.current_target_bbox = selected_target['bbox']
                    self.target_state.current_target_center = selected_target['center']
                    
                    self.target_state.previous_positions.append(
                        np.array([selected_target['center'][0], selected_target['center'][1], 0])
                    )
                    
                    target_world = self.motion_controller.pixel_to_world(
                        selected_target['center'], frame, selected_target
                    )
                    self.motion_controller.move_to_target(target_world, frame, selected_target)
                    
                else:
                    if self.occlusion_handler.handle_occlusion():
                        if self.target_state.predicted_position is not None:
                            pass
            else:
                selected_target = None
            
            vis_frame = self.visualizer.draw_tracking_overlay(
                frame, self.target_state, detected_targets, selected_target
            )
            
            cv2.imshow('实时追踪系统', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('用户退出')
                break
            elif key == ord(' '):
                self.target_state.is_tracking = not self.target_state.is_tracking
                status = '开始追踪' if self.target_state.is_tracking else '暂停追踪'
                print(status)
                tts(status)
                play_wav('temp/tts.wav')
        
        self.stop()
    
    def stop(self):
        print('停止实时追踪系统...')
        self.is_running = False
        
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=2)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        self.motion_controller.stop_movement()
        
        back_zero()
        
        print('实时追踪系统已停止')
        tts('实时追踪系统已停止')
        play_wav('temp/tts.wav')

_global_tracker = None

def start_tracking(target_type=None):
    '''
    启动实时追踪系统
    target_type: 指定追踪目标类型，可选值: 'face', 'red_box', 'green_box', 'blue_box', 'yellow_box'
                如果为 None，则自动选择优先级最高的目标
    '''
    global _global_tracker
    
    print(f'启动实时追踪，目标类型: {target_type if target_type else "自动选择"}')
    
    try:
        _global_tracker = RealTimeTracker()
        
        if target_type:
            _global_tracker.multi_target_manager.switch_target(target_type)
            target_name = TrackingConfig.TRACKING_TARGET_TYPES[target_type]['name']
            tts(f'好的，开始追踪{target_name}')
        else:
            tts('好的，开始实时追踪，自动选择目标')
        
        play_wav('temp/tts.wav')
        
        _global_tracker.initialize()
        _global_tracker.start()
        
    except Exception as e:
        print(f'追踪系统错误: {e}')
        import traceback
        traceback.print_exc()

def stop_tracking():
    '''
    停止实时追踪系统
    '''
    global _global_tracker
    
    print('停止实时追踪')
    
    if _global_tracker:
        _global_tracker.stop()
        _global_tracker = None
    else:
        tts('当前没有正在运行的追踪系统')
        play_wav('temp/tts.wav')
