# utils_recorder.py
# 同济子豪兄 2024-5-27
# 任务录像与复盘模块：记录执行过程中的所有关键数据

import os
import json
import time
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

_cv2_available = None
def _get_cv2():
    '''惰性导入cv2，缺失时给出提示但不抛出异常'''
    global _cv2_available
    if _cv2_available is False:
        return None
    if _cv2_available is not None:
        return _cv2_available
    try:
        import cv2
        _cv2_available = cv2
        return cv2
    except ImportError:
        _cv2_available = False
        print('[Recorder] 警告: OpenCV(cv2) 未安装，相机帧保存功能不可用')
        print('[Recorder] 其他记录功能（语音、VLM、动作执行）仍然可用')
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(BASE_DIR, 'task_records')

class TaskStatus(Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

@dataclass
class SpeechRecord:
    '''语音记录'''
    audio_path: str
    text: str
    timestamp: float
    duration: Optional[float] = None

@dataclass
class CameraFrame:
    '''相机关键帧'''
    frame_path: str
    timestamp: float
    description: str
    camera_coords: Optional[Dict] = None

@dataclass
class VLMResult:
    '''VLM识别结果'''
    task_type: str
    prompt: str
    raw_response: str
    parsed_result: Dict
    timestamp: float
    image_path: str
    viz_path: Optional[str] = None
    model_used: Optional[str] = None

@dataclass
class ActionPlan:
    '''规划的动作序列'''
    original_instruction: str
    llm_response: Dict
    function_calls: List[str]
    timestamp: float
    agent_response: str

@dataclass
class ActionExecution:
    '''实际执行的动作'''
    function_name: str
    parameters: Dict
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = 'pending'
    joint_angles_before: Optional[List[float]] = None
    joint_angles_after: Optional[List[float]] = None
    coords_before: Optional[List[float]] = None
    coords_after: Optional[List[float]] = None
    error_message: Optional[str] = None
    return_value: Optional[Any] = None

@dataclass
class TaskRecord:
    '''完整的任务记录'''
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = TaskStatus.PENDING.value
    speech_records: List[SpeechRecord] = None
    camera_frames: List[CameraFrame] = None
    vlm_results: List[VLMResult] = None
    action_plan: Optional[ActionPlan] = None
    action_executions: List[ActionExecution] = None
    error_summary: Optional[str] = None
    total_duration: Optional[float] = None

class TaskRecorder:
    '''任务记录器'''
    
    _instance = None
    _current_task: Optional[TaskRecord] = None
    _task_dir: Optional[str] = None
    _task_ended: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        os.makedirs(RECORD_DIR, exist_ok=True)
    
    def start_task(self) -> str:
        '''开始新任务'''
        timestamp = time.time()
        task_id = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
        
        self._task_ended = False
        self._task_dir = os.path.join(RECORD_DIR, task_id)
        os.makedirs(self._task_dir, exist_ok=True)
        os.makedirs(os.path.join(self._task_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self._task_dir, 'audio'), exist_ok=True)
        
        self._current_task = TaskRecord(
            task_id=task_id,
            start_time=timestamp,
            status=TaskStatus.RUNNING.value,
            speech_records=[],
            camera_frames=[],
            vlm_results=[],
            action_executions=[]
        )
        
        print(f'[Recorder] 任务记录已启动: {task_id}')
        return task_id
    
    def record_speech(self, audio_path: str, text: str, duration: Optional[float] = None) -> SpeechRecord:
        '''记录语音'''
        if not self._current_task:
            return None
        
        saved_audio_path = os.path.join(self._task_dir, 'audio', f'speech_{len(self._current_task.speech_records)}.wav')
        if os.path.exists(audio_path):
            shutil.copy2(audio_path, saved_audio_path)
        
        record = SpeechRecord(
            audio_path=os.path.relpath(saved_audio_path, self._task_dir),
            text=text,
            timestamp=time.time(),
            duration=duration
        )
        self._current_task.speech_records.append(record)
        print(f'[Recorder] 语音已记录: {text[:30]}...')
        return record
    
    def record_camera_frame(self, frame, description: str, camera_coords: Optional[Dict] = None) -> CameraFrame:
        '''记录相机关键帧'''
        if not self._current_task:
            return None
        
        frame_idx = len(self._current_task.camera_frames)
        frame_path = os.path.join(self._task_dir, 'images', f'frame_{frame_idx:03d}.jpg')
        
        if isinstance(frame, str):
            if os.path.exists(frame):
                shutil.copy2(frame, frame_path)
            else:
                print(f'[Recorder] 警告: 图像文件不存在: {frame}')
                return None
        else:
            cv2 = _get_cv2()
            if cv2 is None:
                print(f'[Recorder] 警告: 无法保存相机帧（OpenCV未安装）')
                return None
            cv2.imwrite(frame_path, frame)
        
        record = CameraFrame(
            frame_path=os.path.relpath(frame_path, self._task_dir),
            timestamp=time.time(),
            description=description,
            camera_coords=camera_coords
        )
        self._current_task.camera_frames.append(record)
        print(f'[Recorder] 相机帧已记录: {description}')
        return record
    
    def record_vlm_result(self, task_type: str, prompt: str, raw_response: str, 
                          parsed_result: Dict, image_path: str, viz_path: Optional[str] = None,
                          model_used: Optional[str] = None) -> VLMResult:
        '''记录VLM识别结果'''
        if not self._current_task:
            return None
        
        saved_img_path = os.path.join(self._task_dir, 'images', f'vlm_input_{len(self._current_task.vlm_results)}.jpg')
        if os.path.exists(image_path):
            shutil.copy2(image_path, saved_img_path)
        
        saved_viz_path = None
        if viz_path and os.path.exists(viz_path):
            saved_viz_path = os.path.join(self._task_dir, 'images', f'vlm_viz_{len(self._current_task.vlm_results)}.jpg')
            shutil.copy2(viz_path, saved_viz_path)
        
        record = VLMResult(
            task_type=task_type,
            prompt=prompt,
            raw_response=raw_response,
            parsed_result=parsed_result,
            timestamp=time.time(),
            image_path=os.path.relpath(saved_img_path, self._task_dir),
            viz_path=os.path.relpath(saved_viz_path, self._task_dir) if saved_viz_path else None,
            model_used=model_used
        )
        self._current_task.vlm_results.append(record)
        print(f'[Recorder] VLM结果已记录: {task_type}')
        return record
    
    def record_action_plan(self, original_instruction: str, llm_response: Dict, 
                           function_calls: List[str], agent_response: str) -> ActionPlan:
        '''记录动作规划'''
        if not self._current_task:
            return None
        
        plan = ActionPlan(
            original_instruction=original_instruction,
            llm_response=llm_response,
            function_calls=function_calls,
            timestamp=time.time(),
            agent_response=agent_response
        )
        self._current_task.action_plan = plan
        print(f'[Recorder] 动作规划已记录: {len(function_calls)} 个动作')
        return plan
    
    def start_action_execution(self, function_name: str, parameters: Dict,
                                joint_angles_before: Optional[List[float]] = None,
                                coords_before: Optional[List[float]] = None) -> ActionExecution:
        '''开始记录动作执行'''
        if not self._current_task:
            return None
        
        execution = ActionExecution(
            function_name=function_name,
            parameters=parameters,
            start_time=time.time(),
            status='running',
            joint_angles_before=joint_angles_before,
            coords_before=coords_before
        )
        self._current_task.action_executions.append(execution)
        print(f'[Recorder] 开始执行: {function_name}')
        return execution
    
    def end_action_execution(self, function_name: str, status: str = 'success',
                             joint_angles_after: Optional[List[float]] = None,
                             coords_after: Optional[List[float]] = None,
                             error_message: Optional[str] = None,
                             return_value: Optional[Any] = None):
        '''结束记录动作执行'''
        if not self._current_task:
            return
        
        for exec_record in reversed(self._current_task.action_executions):
            if exec_record.function_name == function_name and exec_record.status == 'running':
                exec_record.end_time = time.time()
                exec_record.duration = exec_record.end_time - exec_record.start_time
                exec_record.status = status
                exec_record.joint_angles_after = joint_angles_after
                exec_record.coords_after = coords_after
                exec_record.error_message = error_message
                exec_record.return_value = str(return_value) if return_value else None
                print(f'[Recorder] 执行完成: {function_name}, 耗时: {exec_record.duration:.2f}s, 状态: {status}')
                return
    
    def end_task(self, status: str = TaskStatus.SUCCESS.value, error_summary: Optional[str] = None):
        '''结束任务（防重复调用）'''
        if self._task_ended:
            print(f'[Recorder] 任务已结束，跳过重复调用')
            return
        
        if not self._current_task:
            return
        
        self._task_ended = True
        self._current_task.end_time = time.time()
        self._current_task.status = status
        self._current_task.error_summary = error_summary
        self._current_task.total_duration = self._current_task.end_time - self._current_task.start_time
        
        self._save_to_json()
        print(f'[Recorder] 任务记录已保存: {self._current_task.task_id}')
        print(f'[Recorder] 总耗时: {self._current_task.total_duration:.2f}s, 状态: {status}')
    
    def _save_to_json(self):
        '''保存为JSON文件'''
        if not self._current_task or not self._task_dir:
            return
        
        json_path = os.path.join(self._task_dir, 'task_record.json')
        
        record_dict = asdict(self._current_task)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(record_dict, f, ensure_ascii=False, indent=2, default=str)
        
        print(f'[Recorder] 任务数据已保存至: {json_path}')
    
    def get_current_task_id(self) -> Optional[str]:
        '''获取当前任务ID'''
        return self._current_task.task_id if self._current_task else None
    
    def get_task_dir(self) -> Optional[str]:
        '''获取当前任务目录'''
        return self._task_dir

recorder = TaskRecorder()

def get_recorder() -> TaskRecorder:
    '''获取记录器实例'''
    return recorder
