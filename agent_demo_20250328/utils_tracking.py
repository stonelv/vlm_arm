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
from API_KEY import *

VLM_TRACKING_PROMPT = '''
分析这张图片，找出所有的人脸和指定颜色的物体（红色、绿色、蓝色、黄色盒子/物体）。

对于每个检测到的目标，输出以下格式的JSON数据：
{
  "detected": [
    {
      "type": "face",
      "name": "人脸",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ]
}

其中：
- type 可以是: "face" (人脸), "red_box" (红色物体), "green_box" (绿色物体), "blue_box" (蓝色物体), "yellow_box" (黄色物体)
- bbox 是左上角和右下角的像素坐标 [x1, y1, x2, y2]
- confidence 是检测置信度 0.0-1.0

只输出JSON本身，不要输出其它内容，不要包含```json标记。
'''

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
    
    VLM_INTERVAL = 10
    FOCAL_LENGTH = 500.0
    REAL_FACE_WIDTH = 0.16
    REAL_BOX_WIDTH = 0.06
    
    CALI_1_IM = [130, 290]
    CALI_1_MC = [-21.8, -197.4]
    CALI_2_IM = [640, 0]
    CALI_2_MC = [215, -59.1]

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
        self.tracking_config = TrackingConfig()
        self.vlm_frame_counter = 0
        self.vlm_cached_results = None
        self.temp_img_path = 'temp/vl_tracking.jpg'
        
    def _call_vlm_for_detection(self, frame):
        try:
            cv2.imwrite(self.temp_img_path, frame)
            
            import openai
            from openai import OpenAI
            import base64
            
            API_BASE = "https://api.lingyiwanwu.com/v1"
            API_KEY = YI_KEY
            
            client = OpenAI(api_key=API_KEY, base_url=API_BASE)
            
            with open(self.temp_img_path, 'rb') as image_file:
                image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')
            
            completion = client.chat.completions.create(
                model="yi-vision",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": VLM_TRACKING_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image
                                }
                            }
                        ]
                    }
                ]
            )
            
            result_str = completion.choices[0].message.content.strip()
            print(f'VLM 原始返回: {result_str}')
            
            try:
                result = eval(result_str)
                self.vlm_cached_results = result
                return result
            except Exception as e:
                print(f'VLM 结果解析错误: {e}')
                import json
                try:
                    json_start = result_str.find('{')
                    json_end = result_str.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = result_str[json_start:json_end]
                        result = json.loads(json_str)
                        self.vlm_cached_results = result
                        return result
                except:
                    pass
                return None
                
        except Exception as e:
            print(f'VLM 调用错误: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_vlm_results(self, vlm_result, frame):
        detected_targets = []
        
        if not vlm_result or 'detected' not in vlm_result:
            return detected_targets
        
        img_h, img_w = frame.shape[:2]
        
        for item in vlm_result.get('detected', []):
            target_type = item.get('type')
            bbox = item.get('bbox', [])
            name = item.get('name', '')
            
            if target_type not in self.tracking_config.TRACKING_TARGET_TYPES:
                continue
            
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = bbox
            
            if x1 <= 1 and x2 <= 1 and y1 <= 1 and y2 <= 1:
                x1 = int(x1 * img_w)
                y1 = int(y1 * img_h)
                x2 = int(x2 * img_w)
                y2 = int(y2 * img_h)
            
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            
            detected_targets.append({
                'bbox': [x1, y1, x2, y2],
                'center': (center_x, center_y),
                'area': area,
                'type': target_type,
                'confidence': item.get('confidence', 0.8),
                'source': 'vlm'
            })
        
        return detected_targets
    
    def _fallback_detect_color_object(self, frame, target_type):
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
                    'confidence': 0.6,
                    'source': 'cv',
                    'mask': mask
                }
        return None
    
    def detect_face(self, frame):
        self.vlm_frame_counter += 1
        
        if self.vlm_frame_counter % self.tracking_config.VLM_INTERVAL == 1 or not self.vlm_cached_results:
            self._call_vlm_for_detection(frame)
        
        targets = self._parse_vlm_results(self.vlm_cached_results, frame)
        
        for t in targets:
            if t['type'] == 'face':
                return t
        
        return None
    
    def detect_color_object(self, frame, target_type):
        self.vlm_frame_counter += 1
        
        if self.vlm_frame_counter % self.tracking_config.VLM_INTERVAL == 1 or not self.vlm_cached_results:
            self._call_vlm_for_detection(frame)
        
        targets = self._parse_vlm_results(self.vlm_cached_results, frame)
        
        for t in targets:
            if t['type'] == target_type:
                return t
        
        fallback = self._fallback_detect_color_object(frame, target_type)
        return fallback
    
    def detect_all_targets(self, frame):
        self.vlm_frame_counter += 1
        
        vlm_targets = []
        if self.vlm_frame_counter % self.tracking_config.VLM_INTERVAL == 1 or not self.vlm_cached_results:
            print('调用 VLM 进行目标检测...')
            self._call_vlm_for_detection(frame)
        
        if self.vlm_cached_results:
            vlm_targets = self._parse_vlm_results(self.vlm_cached_results, frame)
        
        vlm_types = set([t['type'] for t in vlm_targets])
        
        cv_targets = []
        for target_type in self.tracking_config.TRACKING_TARGET_TYPES:
            if target_type not in vlm_types:
                if target_type != 'face':
                    result = self._fallback_detect_color_object(frame, target_type)
                    if result:
                        cv_targets.append(result)
        
        all_targets = vlm_targets + cv_targets
        
        all_targets.sort(
            key=lambda x: (
                self.tracking_config.TRACKING_TARGET_TYPES[x['type']]['priority'],
                x.get('confidence', 0.5)
            ),
            reverse=True
        )
        
        return all_targets

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
    
    def _estimate_depth_by_area(self, detected_target):
        target_type = detected_target['type']
        area = detected_target.get('area', 10000)
        
        bbox = detected_target.get('bbox', [0, 0, 100, 100])
        pixel_width = bbox[2] - bbox[0]
        pixel_height = bbox[3] - bbox[1]
        
        if target_type == 'face':
            real_width = self.config.REAL_FACE_WIDTH * 1000
        else:
            real_width = self.config.REAL_BOX_WIDTH * 1000
        
        if pixel_width > 0:
            distance = (real_width * self.config.FOCAL_LENGTH) / pixel_width
        else:
            distance = self.config.DESIRED_DISTANCE
        
        distance = max(50, min(400, distance))
        return distance
    
    def _pixel_to_robot_eye2hand(self, pixel_x, pixel_y):
        X_cali_im = [self.config.CALI_1_IM[0], self.config.CALI_2_IM[0]]
        X_cali_mc = [self.config.CALI_1_MC[0], self.config.CALI_2_MC[0]]
        
        Y_cali_im = [self.config.CALI_2_IM[1], self.config.CALI_1_IM[1]]
        Y_cali_mc = [self.config.CALI_2_MC[1], self.config.CALI_1_MC[1]]
        
        X_mc = int(np.interp(pixel_x, X_cali_im, X_cali_mc))
        Y_mc = int(np.interp(pixel_y, Y_cali_im, Y_cali_mc))
        
        return X_mc, Y_mc
    
    def _pixel_to_robot_relative(self, pixel_center, current_coords):
        img_center_x = self.config.CAMERA_WIDTH // 2
        img_center_y = self.config.CAMERA_HEIGHT // 2
        
        offset_x = pixel_center[0] - img_center_x
        offset_y = pixel_center[1] - img_center_y
        
        scale_factor = 0.25
        
        delta_x = -offset_y * scale_factor
        delta_y = offset_x * scale_factor
        
        return delta_x, delta_y
    
    def pixel_to_world(self, pixel_center, frame, detected_target):
        img_h, img_w = frame.shape[:2]
        
        distance = self._estimate_depth_by_area(detected_target)
        print(f'估计目标距离: {distance:.1f} mm')
        
        robot_x, robot_y = self._pixel_to_robot_eye2hand(pixel_center[0], pixel_center[1])
        
        delta_x, delta_y = self._pixel_to_robot_relative(pixel_center, self.current_coords)
        
        desired_x = (robot_x + self.current_coords[0] + delta_x) / 2
        desired_y = (robot_y + self.current_coords[1] + delta_y) / 2
        
        distance_error = distance - self.config.DESIRED_DISTANCE
        if abs(distance_error) > self.config.DISTANCE_TOLERANCE:
            desired_x += distance_error * 0.3
        
        img_center_y = img_h // 2
        offset_y = pixel_center[1] - img_center_y
        height_adjustment = 0
        
        if offset_y < -30:
            height_adjustment = min(8, abs(offset_y) * 0.1)
        elif offset_y > 30:
            height_adjustment = -min(8, abs(offset_y) * 0.1)
        
        desired_z = self.current_coords[2] + height_adjustment
        
        desired_x = np.clip(desired_x, 100, 350)
        desired_y = np.clip(desired_y, -200, 200)
        desired_z = np.clip(desired_z, 50, 250)
        
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
        try:
            self.mc.release_all_servos()
        except:
            pass
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
            '红色物体': 'red_box',
            '绿色': 'green_box',
            '绿色盒子': 'green_box',
            '绿盒子': 'green_box',
            '绿色物体': 'green_box',
            '蓝色': 'blue_box',
            '蓝色盒子': 'blue_box',
            '蓝盒子': 'blue_box',
            '蓝色物体': 'blue_box',
            '黄色': 'yellow_box',
            '黄色盒子': 'yellow_box',
            '黄盒子': 'yellow_box',
            '黄色物体': 'yellow_box',
            '人脸': 'face',
            '脸': 'face',
            '人': 'face',
            '那个人': 'face',
        }
        
        self.action_keywords = {
            '跟着': 'follow',
            '追踪': 'follow',
            '跟踪': 'follow',
            '看': 'follow',
            '注视': 'follow',
            '开始': 'start',
            '启动': 'start',
            '停止': 'stop',
            '结束': 'stop',
            '暂停': 'pause',
            '自动': 'auto',
            '恢复': 'auto',
            '切换到自动': 'auto',
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
            import traceback
            traceback.print_exc()
        
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
        
        tts('我没有理解您的指令，请再说一遍')
        play_wav('temp/tts.wav')
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
            source = target.get('source', 'cv')
            confidence = target.get('confidence', 0.5)
            
            color = (0, 255, 0) if target == selected_target else (128, 128, 128)
            thickness = 3 if target == selected_target else 1
            
            cv2.rectangle(output_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            label = f'{name} ({source}) {confidence:.0%}'
            cv2.putText(output_frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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
        
        cv2.putText(output_frame, 'VLM每10帧检测一次 | 颜色实时检测', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
        
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
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
        print('  - 按 v 键强制触发一次 VLM 检测')
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
            elif key == ord('v'):
                print('强制触发 VLM 检测...')
                self.visual_detector._call_vlm_for_detection(frame)
        
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
