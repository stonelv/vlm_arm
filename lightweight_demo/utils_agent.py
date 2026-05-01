import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class AgentPlanner:
    def __init__(self, config):
        self.config = config
    
    def generate_execution_plan(
        self, 
        task_analysis: Dict[str, Any], 
        image_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = task_analysis.get('result', {})
        task_type = result.get('task_type', 'other')
        
        if task_type == 'pick_place' or task_type == 'move':
            return self._generate_pick_place_plan(result, image_info)
        elif task_type == 'inspect':
            return self._generate_inspect_plan(result, image_info)
        else:
            return self._generate_generic_plan(result, image_info)
    
    def _generate_pick_place_plan(
        self, 
        result: Dict[str, Any], 
        image_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_obj = result.get('start_object', {})
        end_obj = result.get('end_object', {})
        
        img_width = image_info.get('width', 640)
        img_height = image_info.get('height', 480)
        
        keyframes = []
        
        keyframes.append({
            'frame_id': 0,
            'type': 'initial',
            'description': '初始位置 - 机械臂归零',
            'pose': {
                'joints': [0, 0, 0, 0, 0, 0],
                'x': 0,
                'y': 0,
                'z': 200
            },
            'gripper': 'open',
            'duration': 2.0
        })
        
        if start_obj and start_obj.get('center'):
            start_center = start_obj.get('center', [img_width//2, img_height//2])
            start_x, start_y = self._pixel_to_robot(start_center[0], start_center[1], img_width, img_height)
            
            keyframes.append({
                'frame_id': 1,
                'type': 'approach',
                'description': f'接近起始物体: {start_obj.get("name", "未知物体")}',
                'pose': {
                    'joints': None,
                    'x': start_x,
                    'y': start_y,
                    'z': 150
                },
                'gripper': 'open',
                'duration': 1.5,
                'object_info': {
                    'name': start_obj.get('name'),
                    'confidence': start_obj.get('confidence'),
                    'bbox': start_obj.get('bbox')
                }
            })
            
            keyframes.append({
                'frame_id': 2,
                'type': 'descend',
                'description': '下降到抓取位置',
                'pose': {
                    'joints': None,
                    'x': start_x,
                    'y': start_y,
                    'z': 80
                },
                'gripper': 'open',
                'duration': 1.0
            })
            
            keyframes.append({
                'frame_id': 3,
                'type': 'grasp',
                'description': '关闭夹爪，抓取物体',
                'pose': {
                    'joints': None,
                    'x': start_x,
                    'y': start_y,
                    'z': 80
                },
                'gripper': 'close',
                'duration': 0.5
            })
            
            keyframes.append({
                'frame_id': 4,
                'type': 'lift',
                'description': '抬起物体',
                'pose': {
                    'joints': None,
                    'x': start_x,
                    'y': start_y,
                    'z': 180
                },
                'gripper': 'close',
                'duration': 1.0
            })
        
        if end_obj and end_obj.get('center'):
            end_center = end_obj.get('center', [img_width//2, img_height//2])
            end_x, end_y = self._pixel_to_robot(end_center[0], end_center[1], img_width, img_height)
            
            keyframes.append({
                'frame_id': 5,
                'type': 'move',
                'description': f'移动到目标位置: {end_obj.get("name", "目标位置")}',
                'pose': {
                    'joints': None,
                    'x': end_x,
                    'y': end_y,
                    'z': 180
                },
                'gripper': 'close',
                'duration': 2.0,
                'target_info': {
                    'name': end_obj.get('name'),
                    'confidence': end_obj.get('confidence'),
                    'bbox': end_obj.get('bbox')
                }
            })
            
            keyframes.append({
                'frame_id': 6,
                'type': 'descend',
                'description': '下降到放置位置',
                'pose': {
                    'joints': None,
                    'x': end_x,
                    'y': end_y,
                    'z': 90
                },
                'gripper': 'close',
                'duration': 1.0
            })
            
            keyframes.append({
                'frame_id': 7,
                'type': 'release',
                'description': '打开夹爪，放置物体',
                'pose': {
                    'joints': None,
                    'x': end_x,
                    'y': end_y,
                    'z': 90
                },
                'gripper': 'open',
                'duration': 0.5
            })
            
            keyframes.append({
                'frame_id': 8,
                'type': 'retreat',
                'description': '后退离开',
                'pose': {
                    'joints': None,
                    'x': end_x,
                    'y': end_y,
                    'z': 180
                },
                'gripper': 'open',
                'duration': 1.0
            })
        
        keyframes.append({
            'frame_id': len(keyframes),
            'type': 'final',
            'description': '任务完成，返回安全位置',
            'pose': {
                'joints': [0, 0, 0, 0, 0, 0],
                'x': 0,
                'y': 0,
                'z': 200
            },
            'gripper': 'open',
            'duration': 2.0
        })
        
        total_duration = sum(kf.get('duration', 1.0) for kf in keyframes)
        
        return {
            'success': True,
            'plan_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'task_type': result.get('task_type', 'pick_place'),
            'action_description': result.get('action_description', '执行抓取放置任务'),
            'total_keyframes': len(keyframes),
            'estimated_duration': round(total_duration, 1),
            'keyframes': keyframes,
            'summary': self._generate_plan_summary(keyframes, start_obj, end_obj),
            'confidence': result.get('confidence', 0.8)
        }
    
    def _generate_inspect_plan(
        self, 
        result: Dict[str, Any], 
        image_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        keyframes = []
        
        keyframes.append({
            'frame_id': 0,
            'type': 'initial',
            'description': '初始位置',
            'pose': {'joints': [0, 0, 0, 0, 0, 0], 'x': 0, 'y': 0, 'z': 200},
            'gripper': 'open',
            'duration': 2.0
        })
        
        keyframes.append({
            'frame_id': 1,
            'type': 'inspect',
            'description': '移动到观察位置',
            'pose': {'joints': None, 'x': 150, 'y': 0, 'z': 150},
            'gripper': 'open',
            'duration': 1.5
        })
        
        keyframes.append({
            'frame_id': 2,
            'type': 'final',
            'description': '检查完成',
            'pose': {'joints': [0, 0, 0, 0, 0, 0], 'x': 0, 'y': 0, 'z': 200},
            'gripper': 'open',
            'duration': 1.0
        })
        
        total_duration = sum(kf.get('duration', 1.0) for kf in keyframes)
        
        return {
            'success': True,
            'plan_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'task_type': 'inspect',
            'action_description': result.get('action_description', '执行检查任务'),
            'total_keyframes': len(keyframes),
            'estimated_duration': round(total_duration, 1),
            'keyframes': keyframes,
            'summary': '检查任务：移动到观察位置进行视觉检查',
            'confidence': result.get('confidence', 0.8)
        }
    
    def _generate_generic_plan(
        self, 
        result: Dict[str, Any], 
        image_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        keyframes = [{
            'frame_id': 0,
            'type': 'initial',
            'description': '等待指令',
            'pose': {'joints': [0, 0, 0, 0, 0, 0], 'x': 0, 'y': 0, 'z': 200},
            'gripper': 'open',
            'duration': 1.0
        }]
        
        return {
            'success': True,
            'plan_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'task_type': result.get('task_type', 'other'),
            'action_description': result.get('action_description', '等待进一步指令'),
            'total_keyframes': len(keyframes),
            'estimated_duration': 1.0,
            'keyframes': keyframes,
            'summary': '当前任务类型不支持自动规划，请确认指令',
            'confidence': result.get('confidence', 0.5)
        }
    
    def _pixel_to_robot(self, px: int, py: int, img_width: int, img_height: int) -> tuple:
        x_range = (-180, 180)
        y_range = (-180, 180)
        
        x = x_range[0] + (px / img_width) * (x_range[1] - x_range[0])
        y = y_range[0] + (py / img_height) * (y_range[1] - y_range[0])
        
        return round(x, 1), round(y, 1)
    
    def _generate_plan_summary(
        self, 
        keyframes: List[Dict], 
        start_obj: Dict, 
        end_obj: Dict
    ) -> str:
        start_name = start_obj.get('name', '起始物体') if start_obj else '起始位置'
        end_name = end_obj.get('name', '目标物体') if end_obj else '目标位置'
        
        grasp_count = sum(1 for kf in keyframes if kf.get('type') == 'grasp')
        release_count = sum(1 for kf in keyframes if kf.get('type') == 'release')
        
        return (f"计划执行 {len(keyframes)} 个关键帧动作。"
                f"从 {start_name} 抓取物体 ({grasp_count} 次)，"
                f"移动到 {end_name} 并放置 ({release_count} 次)。")
    
    def validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        issues = []
        warnings = []
        
        if not plan.get('success'):
            issues.append("计划生成失败")
        
        keyframes = plan.get('keyframes', [])
        if len(keyframes) == 0:
            issues.append("没有定义任何关键帧")
        
        for i, kf in enumerate(keyframes):
            if kf.get('gripper') not in ['open', 'close']:
                warnings.append(f"关键帧 {i}: 夹爪状态未定义")
            
            pose = kf.get('pose', {})
            if pose.get('joints') is None and (pose.get('x') is None or pose.get('y') is None):
                warnings.append(f"关键帧 {i}: 位置信息不完整")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'can_execute': len(issues) == 0
        }
