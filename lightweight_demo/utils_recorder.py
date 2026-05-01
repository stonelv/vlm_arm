import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class DialogueType(Enum):
    USER_INPUT = "user_input"
    SYSTEM_RESPONSE = "system_response"
    VLM_RESULT = "vlm_result"
    PLAN_UPDATE = "plan_update"
    EXECUTION_LOG = "execution_log"

@dataclass
class DialogueEntry:
    timestamp: str
    type: str
    content: Dict[str, Any]
    image_path: Optional[str] = None

@dataclass
class KeyframeRecord:
    frame_id: int
    type: str
    description: str
    pose: Dict[str, Any]
    gripper: str
    duration: float
    executed: bool
    execution_timestamp: Optional[str]
    actual_pose: Optional[Dict[str, Any]]

@dataclass
class TaskRecord:
    task_id: str
    created_at: str
    updated_at: str
    status: str
    user_instruction: str
    image_path: Optional[str]
    detections: List[Dict[str, Any]]
    plan: Optional[Dict[str, Any]]
    keyframes: List[KeyframeRecord]
    dialogue_history: List[DialogueEntry]
    execution_log: List[Dict[str, Any]]
    final_status: Optional[str]
    exported_at: Optional[str]

class TaskRecorder:
    def __init__(self, record_dir='records'):
        self.record_dir = record_dir
        self.current_task: Optional[TaskRecord] = None
        os.makedirs(record_dir, exist_ok=True)
    
    def create_task(self, user_instruction: str, image_path: str = None) -> str:
        task_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.current_task = TaskRecord(
            task_id=task_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status=TaskStatus.PENDING.value,
            user_instruction=user_instruction,
            image_path=image_path,
            detections=[],
            plan=None,
            keyframes=[],
            dialogue_history=[],
            execution_log=[],
            final_status=None,
            exported_at=None
        )
        
        self._add_dialogue(
            DialogueType.USER_INPUT,
            {"instruction": user_instruction},
            image_path
        )
        
        return task_id
    
    def update_status(self, status: TaskStatus):
        if self.current_task:
            self.current_task.status = status.value
            self.current_task.updated_at = datetime.now().isoformat()
    
    def add_detections(self, detections: List[Dict[str, Any]], raw_response: str = None):
        if self.current_task:
            self.current_task.detections = detections
            self._add_dialogue(
                DialogueType.VLM_RESULT,
                {
                    "detections": detections,
                    "raw_response": raw_response
                }
            )
    
    def set_plan(self, plan: Dict[str, Any]):
        if self.current_task:
            self.current_task.plan = plan
            self.current_task.keyframes = [
                KeyframeRecord(
                    frame_id=kf['frame_id'],
                    type=kf['type'],
                    description=kf['description'],
                    pose=kf['pose'],
                    gripper=kf['gripper'],
                    duration=kf['duration'],
                    executed=False,
                    execution_timestamp=None,
                    actual_pose=None
                ) for kf in plan.get('keyframes', [])
            ]
            self._add_dialogue(
                DialogueType.PLAN_UPDATE,
                {
                    "plan_id": plan.get('plan_id'),
                    "total_keyframes": plan.get('total_keyframes'),
                    "estimated_duration": plan.get('estimated_duration'),
                    "summary": plan.get('summary')
                }
            )
    
    def record_keyframe_execution(
        self, 
        frame_id: int, 
        actual_pose: Dict[str, Any] = None,
        success: bool = True
    ):
        if self.current_task:
            for kf in self.current_task.keyframes:
                if kf.frame_id == frame_id:
                    kf.executed = True
                    kf.execution_timestamp = datetime.now().isoformat()
                    kf.actual_pose = actual_pose
                    
                    self.current_task.execution_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "frame_id": frame_id,
                        "success": success,
                        "actual_pose": actual_pose
                    })
                    break
    
    def _add_dialogue(self, dialogue_type: DialogueType, content: Dict[str, Any], image_path: str = None):
        if self.current_task:
            entry = DialogueEntry(
                timestamp=datetime.now().isoformat(),
                type=dialogue_type.value,
                content=content,
                image_path=image_path
            )
            self.current_task.dialogue_history.append(entry)
    
    def add_system_response(self, response: str):
        self._add_dialogue(
            DialogueType.SYSTEM_RESPONSE,
            {"response": response}
        )
    
    def add_execution_log(self, log_entry: Dict[str, Any]):
        if self.current_task:
            log_entry["timestamp"] = datetime.now().isoformat()
            self.current_task.execution_log.append(log_entry)
    
    def finalize_task(self, status: TaskStatus, final_message: str = None):
        if self.current_task:
            self.current_task.final_status = status.value
            self.current_task.status = status.value
            self.current_task.updated_at = datetime.now().isoformat()
            
            if final_message:
                self.add_system_response(final_message)
    
    def get_task_summary(self) -> Dict[str, Any]:
        if not self.current_task:
            return {"error": "No active task"}
        
        return {
            "task_id": self.current_task.task_id,
            "status": self.current_task.status,
            "user_instruction": self.current_task.user_instruction,
            "created_at": self.current_task.created_at,
            "total_keyframes": len(self.current_task.keyframes),
            "executed_keyframes": sum(1 for kf in self.current_task.keyframes if kf.executed),
            "dialogue_entries": len(self.current_task.dialogue_history),
            "detections_count": len(self.current_task.detections)
        }
    
    def export_task(self, format: str = 'json') -> str:
        if not self.current_task:
            raise ValueError("No active task to export")
        
        self.current_task.exported_at = datetime.now().isoformat()
        
        export_data = {
            "task_id": self.current_task.task_id,
            "created_at": self.current_task.created_at,
            "updated_at": self.current_task.updated_at,
            "exported_at": self.current_task.exported_at,
            "status": self.current_task.status,
            "final_status": self.current_task.final_status,
            "user_instruction": self.current_task.user_instruction,
            "image_path": self.current_task.image_path,
            "detections": self.current_task.detections,
            "plan": self.current_task.plan,
            "keyframes": [asdict(kf) for kf in self.current_task.keyframes],
            "dialogue_history": [asdict(de) for de in self.current_task.dialogue_history],
            "execution_log": self.current_task.execution_log,
            "summary": self.get_task_summary()
        }
        
        if format == 'json':
            filename = f"task_{self.current_task.task_id}.json"
            filepath = os.path.join(self.record_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return filepath
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def load_task(self, filepath: str) -> TaskRecord:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        keyframes = [
            KeyframeRecord(**kf) for kf in data.get('keyframes', [])
        ]
        
        dialogue_history = [
            DialogueEntry(**de) for de in data.get('dialogue_history', [])
        ]
        
        self.current_task = TaskRecord(
            task_id=data['task_id'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            status=data['status'],
            user_instruction=data['user_instruction'],
            image_path=data.get('image_path'),
            detections=data.get('detections', []),
            plan=data.get('plan'),
            keyframes=keyframes,
            dialogue_history=dialogue_history,
            execution_log=data.get('execution_log', []),
            final_status=data.get('final_status'),
            exported_at=data.get('exported_at')
        )
        
        return self.current_task
    
    def list_all_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        for filename in os.listdir(self.record_dir):
            if filename.startswith('task_') and filename.endswith('.json'):
                filepath = os.path.join(self.record_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        tasks.append({
                            "task_id": data.get('task_id'),
                            "created_at": data.get('created_at'),
                            "status": data.get('status'),
                            "user_instruction": data.get('user_instruction', '')[:50],
                            "filepath": filepath
                        })
                except:
                    continue
        
        return sorted(tasks, key=lambda x: x['created_at'], reverse=True)
