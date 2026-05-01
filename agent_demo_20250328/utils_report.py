# utils_report.py
# 同济子豪兄 2024-5-27
# 任务复盘报告生成模块

import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
import glob

RECORD_DIR = 'task_records'

def format_timestamp(timestamp: float) -> str:
    '''格式化时间戳'''
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def format_duration(seconds: Optional[float]) -> str:
    '''格式化时长'''
    if seconds is None:
        return 'N/A'
    if seconds < 60:
        return f'{seconds:.2f}秒'
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f'{minutes}分{secs:.2f}秒'

def get_status_icon(status: str) -> str:
    '''获取状态图标'''
    status_icons = {
        'success': '✅',
        'failed': '❌',
        'running': '⏳',
        'pending': '📋',
        'cancelled': '🚫'
    }
    return status_icons.get(status.lower(), '❓')

def generate_markdown_report(task_dir: str, output_path: Optional[str] = None) -> str:
    '''
    生成Markdown格式的任务复盘报告
    
    Args:
        task_dir: 任务记录目录路径
        output_path: 输出报告路径（可选）
    
    Returns:
        生成的报告内容
    '''
    json_path = os.path.join(task_dir, 'task_record.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'任务记录文件不存在: {json_path}')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    
    task_id = record.get('task_id', 'unknown')
    
    md_lines = []
    
    md_lines.append(f'# 任务复盘报告 - {task_id}')
    md_lines.append('')
    md_lines.append(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    md_lines.append('')
    
    md_lines.append('## 📊 任务概览')
    md_lines.append('')
    md_lines.append('| 项目 | 内容 |')
    md_lines.append('|------|------|')
    md_lines.append(f'| 任务ID | {task_id} |')
    md_lines.append(f'| 开始时间 | {format_timestamp(record.get("start_time", 0))} |')
    md_lines.append(f'| 结束时间 | {format_timestamp(record.get("end_time", 0)) if record.get("end_time") else "N/A"} |')
    md_lines.append(f'| 总耗时 | {format_duration(record.get("total_duration"))} |')
    md_lines.append(f'| 任务状态 | {get_status_icon(record.get("status"))} {record.get("status", "N/A")} |')
    md_lines.append('')
    
    if record.get('error_summary'):
        md_lines.append('## ⚠️ 错误摘要')
        md_lines.append('')
        md_lines.append(f'```')
        md_lines.append(record.get('error_summary'))
        md_lines.append('```')
        md_lines.append('')
    
    speech_records = record.get('speech_records', [])
    if speech_records:
        md_lines.append('## 🎤 语音记录')
        md_lines.append('')
        for idx, speech in enumerate(speech_records):
            md_lines.append(f'### 第 {idx+1} 条语音')
            md_lines.append('')
            md_lines.append(f'- **时间**: {format_timestamp(speech.get("timestamp", 0))}')
            md_lines.append(f'- **时长**: {speech.get("duration", "N/A")} 秒')
            md_lines.append(f'- **语音文本**: {speech.get("text", "N/A")}')
            audio_path = speech.get('audio_path', '')
            if audio_path and os.path.exists(os.path.join(task_dir, audio_path)):
                md_lines.append(f'- **音频文件**: `{audio_path}`')
            md_lines.append('')
    
    action_plan = record.get('action_plan')
    if action_plan:
        md_lines.append('## 📝 动作规划')
        md_lines.append('')
        md_lines.append(f'### 原始指令')
        md_lines.append(f'> {action_plan.get("original_instruction", "N/A")}')
        md_lines.append('')
        md_lines.append(f'### 智能体回复')
        md_lines.append(f'> {action_plan.get("agent_response", "N/A")}')
        md_lines.append('')
        md_lines.append(f'### 规划的动作序列')
        md_lines.append('')
        function_calls = action_plan.get('function_calls', [])
        if function_calls:
            for idx, func in enumerate(function_calls):
                md_lines.append(f'{idx+1}. `{func}`')
        else:
            md_lines.append('无规划动作')
        md_lines.append('')
    
    vlm_results = record.get('vlm_results', [])
    if vlm_results:
        md_lines.append('## 👁️ VLM识别结果')
        md_lines.append('')
        for idx, vlm in enumerate(vlm_results):
            md_lines.append(f'### 第 {idx+1} 次VLM调用')
            md_lines.append('')
            md_lines.append(f'- **时间**: {format_timestamp(vlm.get("timestamp", 0))}')
            md_lines.append(f'- **任务类型**: {vlm.get("task_type", "N/A")}')
            md_lines.append(f'- **使用模型**: {vlm.get("model_used", "N/A")}')
            md_lines.append(f'- **提示词**: {vlm.get("prompt", "N/A")}')
            md_lines.append('')
            
            parsed_result = vlm.get('parsed_result', {})
            md_lines.append('#### 解析结果')
            md_lines.append('')
            md_lines.append('```json')
            md_lines.append(json.dumps(parsed_result, ensure_ascii=False, indent=2))
            md_lines.append('```')
            md_lines.append('')
            
            img_path = vlm.get('image_path', '')
            if img_path and os.path.exists(os.path.join(task_dir, img_path)):
                md_lines.append(f'#### 输入图像')
                md_lines.append('')
                md_lines.append(f'![输入图像]({img_path})')
                md_lines.append('')
            
            viz_path = vlm.get('viz_path', '')
            if viz_path and os.path.exists(os.path.join(task_dir, viz_path)):
                md_lines.append(f'#### 可视化结果')
                md_lines.append('')
                md_lines.append(f'![可视化结果]({viz_path})')
                md_lines.append('')
    
    camera_frames = record.get('camera_frames', [])
    if camera_frames:
        md_lines.append('## 📷 相机关键帧')
        md_lines.append('')
        for idx, frame in enumerate(camera_frames):
            md_lines.append(f'### 帧 {idx+1}')
            md_lines.append('')
            md_lines.append(f'- **时间**: {format_timestamp(frame.get("timestamp", 0))}')
            md_lines.append(f'- **描述**: {frame.get("description", "N/A")}')
            md_lines.append('')
            
            frame_path = frame.get('frame_path', '')
            if frame_path and os.path.exists(os.path.join(task_dir, frame_path)):
                md_lines.append(f'![{frame.get("description", "相机帧")}]({frame_path})')
                md_lines.append('')
            
            coords = frame.get('camera_coords')
            if coords:
                md_lines.append('#### 坐标信息')
                md_lines.append('')
                md_lines.append('```json')
                md_lines.append(json.dumps(coords, ensure_ascii=False, indent=2))
                md_lines.append('```')
                md_lines.append('')
    
    action_executions = record.get('action_executions', [])
    if action_executions:
        md_lines.append('## ⚙️ 动作执行详情')
        md_lines.append('')
        
        md_lines.append('### 执行时序')
        md_lines.append('')
        md_lines.append('| 序号 | 动作 | 状态 | 耗时 |')
        md_lines.append('|------|------|------|------|')
        for idx, exec_item in enumerate(action_executions):
            status_icon = get_status_icon(exec_item.get('status', 'unknown'))
            duration_str = format_duration(exec_item.get('duration'))
            md_lines.append(f'| {idx+1} | `{exec_item.get("function_name", "N/A")}` | {status_icon} {exec_item.get("status")} | {duration_str} |')
        md_lines.append('')
        
        md_lines.append('### 详细执行记录')
        md_lines.append('')
        for idx, exec_item in enumerate(action_executions):
            md_lines.append(f'#### 动作 {idx+1}: {exec_item.get("function_name")}')
            md_lines.append('')
            md_lines.append(f'- **开始时间**: {format_timestamp(exec_item.get("start_time", 0))}')
            md_lines.append(f'- **结束时间**: {format_timestamp(exec_item.get("end_time", 0)) if exec_item.get("end_time") else "N/A"}')
            md_lines.append(f'- **耗时**: {format_duration(exec_item.get("duration"))}')
            md_lines.append(f'- **状态**: {get_status_icon(exec_item.get("status"))} {exec_item.get("status")}')
            md_lines.append('')
            
            joints_before = exec_item.get('joint_angles_before')
            joints_after = exec_item.get('joint_angles_after')
            if joints_before or joints_after:
                md_lines.append('##### 关节角度变化')
                md_lines.append('')
                md_lines.append('| 关节 | 执行前 | 执行后 |')
                md_lines.append('|------|--------|--------|')
                for j in range(6):
                    before = f'{joints_before[j]:.2f}°' if joints_before and len(joints_before) > j else 'N/A'
                    after = f'{joints_after[j]:.2f}°' if joints_after and len(joints_after) > j else 'N/A'
                    md_lines.append(f'| {j+1} | {before} | {after} |')
                md_lines.append('')
            
            coords_before = exec_item.get('coords_before')
            coords_after = exec_item.get('coords_after')
            if coords_before or coords_after:
                md_lines.append('##### 末端坐标变化')
                md_lines.append('')
                md_lines.append('| 坐标 | 执行前 | 执行后 |')
                md_lines.append('|------|--------|--------|')
                coord_names = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
                for j in range(6):
                    before = f'{coords_before[j]:.2f}' if coords_before and len(coords_before) > j else 'N/A'
                    after = f'{coords_after[j]:.2f}' if coords_after and len(coords_after) > j else 'N/A'
                    md_lines.append(f'| {coord_names[j]} | {before} | {after} |')
                md_lines.append('')
            
            params = exec_item.get('parameters', {})
            if params:
                md_lines.append('##### 调用参数')
                md_lines.append('')
                md_lines.append('```json')
                md_lines.append(json.dumps(params, ensure_ascii=False, indent=2))
                md_lines.append('```')
                md_lines.append('')
            
            error_msg = exec_item.get('error_message')
            if error_msg:
                md_lines.append('##### 错误信息')
                md_lines.append('')
                md_lines.append('```')
                md_lines.append(error_msg)
                md_lines.append('```')
                md_lines.append('')
            
            ret_val = exec_item.get('return_value')
            if ret_val:
                md_lines.append('##### 返回值')
                md_lines.append('')
                md_lines.append(f'`{ret_val}`')
                md_lines.append('')
    
    md_lines.append('## 📈 统计信息')
    md_lines.append('')
    total_actions = len(action_executions)
    success_actions = sum(1 for e in action_executions if e.get('status') == 'success')
    failed_actions = sum(1 for e in action_executions if e.get('status') == 'failed')
    total_vlm_calls = len(vlm_results)
    total_frames = len(camera_frames)
    
    md_lines.append(f'- **总动作数**: {total_actions}')
    md_lines.append(f'- **成功动作**: {success_actions}')
    md_lines.append(f'- **失败动作**: {failed_actions}')
    md_lines.append(f'- **VLM调用次数**: {total_vlm_calls}')
    md_lines.append(f'- **相机帧数量**: {total_frames}')
    md_lines.append('')
    
    md_lines.append('---')
    md_lines.append(f'*报告生成工具: VLM机械臂任务复盘系统*')
    
    report_content = '\n'.join(md_lines)
    
    if output_path is None:
        output_path = os.path.join(task_dir, 'report.md')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f'[Report] 报告已生成: {output_path}')
    
    return report_content

def list_all_tasks() -> list:
    '''列出所有任务'''
    if not os.path.exists(RECORD_DIR):
        return []
    
    task_dirs = sorted([d for d in os.listdir(RECORD_DIR) 
                       if os.path.isdir(os.path.join(RECORD_DIR, d))],
                      reverse=True)
    
    tasks = []
    for task_dir in task_dirs:
        json_path = os.path.join(RECORD_DIR, task_dir, 'task_record.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                tasks.append({
                    'task_id': record.get('task_id', task_dir),
                    'task_dir': os.path.join(RECORD_DIR, task_dir),
                    'status': record.get('status', 'unknown'),
                    'start_time': format_timestamp(record.get('start_time', 0)),
                    'total_duration': format_duration(record.get('total_duration')),
                    'error_summary': record.get('error_summary')
                })
            except:
                tasks.append({
                    'task_id': task_dir,
                    'task_dir': os.path.join(RECORD_DIR, task_dir),
                    'status': 'error',
                    'start_time': 'N/A',
                    'total_duration': 'N/A',
                    'error_summary': '读取任务记录失败'
                })
    
    return tasks

def generate_latest_report() -> Optional[str]:
    '''生成最近一次任务的报告'''
    tasks = list_all_tasks()
    if not tasks:
        print('[Report] 没有找到任何任务记录')
        return None
    
    latest_task = tasks[0]
    print(f'[Report] 生成最近任务的报告: {latest_task["task_id"]}')
    
    return generate_markdown_report(latest_task['task_dir'])

def generate_all_reports() -> int:
    '''为所有任务生成报告'''
    tasks = list_all_tasks()
    if not tasks:
        print('[Report] 没有找到任何任务记录')
        return 0
    
    count = 0
    for task in tasks:
        try:
            generate_markdown_report(task['task_dir'])
            count += 1
        except Exception as e:
            print(f'[Report] 生成报告失败 {task["task_id"]}: {e}')
    
    print(f'[Report] 已为 {count} 个任务生成报告')
    return count
