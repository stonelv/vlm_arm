# utils_visualizer.py
# 同济子豪兄 2024-5-27
# 任务可视化回放模块

import os
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import glob

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print('[Visualizer] matplotlib未安装，部分可视化功能不可用')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(BASE_DIR, 'task_records')

def load_task_record(task_dir: str) -> Dict:
    '''加载任务记录'''
    json_path = os.path.join(task_dir, 'task_record.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'任务记录文件不存在: {json_path}')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_timeline_chart(task_dir: str, output_path: Optional[str] = None, 
                            show_plot: bool = False) -> Optional[str]:
    '''
    生成任务时间线图表
    
    Args:
        task_dir: 任务目录
        output_path: 输出图片路径
        show_plot: 是否显示图表
    
    Returns:
        输出路径
    '''
    if not MATPLOTLIB_AVAILABLE:
        print('[Visualizer] 需要安装matplotlib: pip install matplotlib')
        return None
    
    record = load_task_record(task_dir)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    action_executions = record.get('action_executions', [])
    
    colors = {
        'success': '#2ecc71',
        'failed': '#e74c3c',
        'running': '#3498db',
        'pending': '#95a5a6',
        'cancelled': '#f39c12'
    }
    
    y_labels = []
    y_positions = []
    
    start_time = record.get('start_time', 0)
    
    for idx, exec_item in enumerate(reversed(action_executions)):
        status = exec_item.get('status', 'unknown')
        color = colors.get(status, '#95a5a6')
        
        exec_start = exec_item.get('start_time', start_time) - start_time
        exec_duration = exec_item.get('duration', 0) or 0
        
        y_labels.append(exec_item.get('function_name', f'action_{idx}'))
        y_positions.append(idx)
        
        ax.barh(idx, exec_duration, left=exec_start, color=color, alpha=0.8, height=0.6)
        
        if exec_duration > 0:
            ax.text(exec_start + exec_duration/2, idx, 
                   f'{exec_duration:.2f}s',
                   ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('动作')
    ax.set_title(f'任务执行时间线 - {record.get("task_id", "unknown")}')
    
    legend_elements = [
        mpatches.Patch(facecolor=colors['success'], label='成功'),
        mpatches.Patch(facecolor=colors['failed'], label='失败'),
        mpatches.Patch(facecolor=colors['cancelled'], label='取消')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(task_dir, 'timeline_chart.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[Visualizer] 时间线图表已保存: {output_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_path

def generate_joint_angle_chart(task_dir: str, output_path: Optional[str] = None,
                                show_plot: bool = False) -> Optional[str]:
    '''
    生成关节角度变化图表
    
    Args:
        task_dir: 任务目录
        output_path: 输出图片路径
        show_plot: 是否显示图表
    
    Returns:
        输出路径
    '''
    if not MATPLOTLIB_AVAILABLE:
        print('[Visualizer] 需要安装matplotlib: pip install matplotlib')
        return None
    
    record = load_task_record(task_dir)
    action_executions = record.get('action_executions', [])
    
    joint_names = ['关节1', '关节2', '关节3', '关节4', '关节5', '关节6']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    action_names = [e.get('function_name', f'action_{i}') for i, e in enumerate(action_executions)]
    x = np.arange(len(action_executions))
    
    for j_idx in range(6):
        ax = axes[j_idx]
        
        before_angles = []
        after_angles = []
        valid_indices = []
        
        for i, exec_item in enumerate(action_executions):
            before = exec_item.get('joint_angles_before')
            after = exec_item.get('joint_angles_after')
            
            if before and len(before) > j_idx:
                before_angles.append(before[j_idx])
                valid_indices.append(i)
            else:
                before_angles.append(None)
            
            if after and len(after) > j_idx:
                after_angles.append(after[j_idx])
            else:
                after_angles.append(None)
        
        width = 0.35
        
        valid_before = [b for b in before_angles if b is not None]
        valid_after = [a for a in after_angles if a is not None]
        valid_x = [i for i, b in enumerate(before_angles) if b is not None]
        
        if valid_before:
            ax.bar([i - width/2 for i in valid_x], valid_before, width, label='执行前', color='#3498db', alpha=0.7)
        
        valid_after_x = [i for i, a in enumerate(after_angles) if a is not None]
        if valid_after:
            ax.bar([i + width/2 for i in valid_after_x], valid_after, width, label='执行后', color='#e74c3c', alpha=0.7)
        
        ax.set_title(joint_names[j_idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('动作')
        ax.set_ylabel('角度 (°)')
        ax.set_xticks(x)
        ax.set_xticklabels(action_names, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'关节角度变化 - {record.get("task_id", "unknown")}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(task_dir, 'joint_angles_chart.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[Visualizer] 关节角度图表已保存: {output_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_path

def generate_coords_chart(task_dir: str, output_path: Optional[str] = None,
                           show_plot: bool = False) -> Optional[str]:
    '''
    生成末端坐标变化图表
    
    Args:
        task_dir: 任务目录
        output_path: 输出图片路径
        show_plot: 是否显示图表
    
    Returns:
        输出路径
    '''
    if not MATPLOTLIB_AVAILABLE:
        print('[Visualizer] 需要安装matplotlib: pip install matplotlib')
        return None
    
    record = load_task_record(task_dir)
    action_executions = record.get('action_executions', [])
    
    coord_names = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    action_names = [e.get('function_name', f'action_{i}') for i, e in enumerate(action_executions)]
    x = np.arange(len(action_executions))
    
    for c_idx in range(6):
        ax = axes[c_idx]
        
        before_coords = []
        after_coords = []
        
        for i, exec_item in enumerate(action_executions):
            before = exec_item.get('coords_before')
            after = exec_item.get('coords_after')
            
            if before and len(before) > c_idx:
                before_coords.append(before[c_idx])
            else:
                before_coords.append(None)
            
            if after and len(after) > c_idx:
                after_coords.append(after[c_idx])
            else:
                after_coords.append(None)
        
        width = 0.35
        
        valid_before = [b for b in before_coords if b is not None]
        valid_after = [a for a in after_coords if a is not None]
        valid_x = [i for i, b in enumerate(before_coords) if b is not None]
        
        if valid_before:
            ax.bar([i - width/2 for i in valid_x], valid_before, width, label='执行前', color='#3498db', alpha=0.7)
        
        valid_after_x = [i for i, a in enumerate(after_coords) if a is not None]
        if valid_after:
            ax.bar([i + width/2 for i in valid_after_x], valid_after, width, label='执行后', color='#e74c3c', alpha=0.7)
        
        unit = 'mm' if c_idx < 3 else '°'
        ax.set_title(f'{coord_names[c_idx]} ({unit})', fontsize=12, fontweight='bold')
        ax.set_xlabel('动作')
        ax.set_ylabel(f'值 ({unit})')
        ax.set_xticks(x)
        ax.set_xticklabels(action_names, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'末端坐标变化 - {record.get("task_id", "unknown")}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(task_dir, 'coords_chart.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[Visualizer] 坐标变化图表已保存: {output_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_path

def generate_all_charts(task_dir: str, show_plot: bool = False) -> Dict[str, str]:
    '''
    生成所有图表
    
    Args:
        task_dir: 任务目录
        show_plot: 是否显示图表
    
    Returns:
        图表路径字典
    '''
    results = {}
    
    try:
        result = generate_timeline_chart(task_dir, show_plot=show_plot)
        if result:
            results['timeline'] = result
    except Exception as e:
        print(f'[Visualizer] 生成时间线图表失败: {e}')
    
    try:
        result = generate_joint_angle_chart(task_dir, show_plot=show_plot)
        if result:
            results['joint_angles'] = result
    except Exception as e:
        print(f'[Visualizer] 生成关节角度图表失败: {e}')
    
    try:
        result = generate_coords_chart(task_dir, show_plot=show_plot)
        if result:
            results['coords'] = result
    except Exception as e:
        print(f'[Visualizer] 生成坐标图表失败: {e}')
    
    return results

def create_playback_script(task_dir: str, output_path: Optional[str] = None) -> str:
    '''
    创建回放脚本
    
    Args:
        task_dir: 任务目录
        output_path: 输出脚本路径
    
    Returns:
        脚本路径
    '''
    record = load_task_record(task_dir)
    task_id = record.get('task_id', 'unknown')
    
    if output_path is None:
        output_path = os.path.join(task_dir, 'playback.py')
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 任务回放脚本 - 任务ID: {task_id}
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import json
import os
import sys

def load_record():
    """加载任务记录"""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_record.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def playback_summary():
    """显示任务摘要"""
    record = load_record()
    
    print("=" * 60)
    print(f"任务回放 - {record.get('task_id', 'unknown')}")
    print("=" * 60)
    
    start_time = record.get('start_time', 0)
    end_time = record.get('end_time')
    total_duration = record.get('total_duration')
    
    print(f"开始时间: {record.get('start_time_formatted', 'N/A')}")
    print(f"结束时间: {record.get('end_time_formatted', 'N/A')}")
    print(f"总耗时: {total_duration:.2f}秒" if total_duration else "总耗时: N/A")
    print(f"任务状态: {record.get('status', 'N/A')}")
    
    if record.get('error_summary'):
        print(f"错误摘要: {record['error_summary']}")
    
    print()
    
    actions = record.get('action_executions', [])
    print(f"动作序列 (共{len(actions)}个动作):")
    print("-" * 60)
    
    for i, action in enumerate(actions, 1):
        status = action.get('status', 'unknown')
        func_name = action.get('function_name', 'unknown')
        duration = action.get('duration')
        error = action.get('error_message')
        
        status_icon = {
            'success': '✅',
            'failed': '❌',
            'running': '⏳',
            'pending': '📋',
            'cancelled': '🚫'
        }.get(status, '❓')
        
        print(f"\\n{i}. {status_icon} {func_name}")
        print(f"   状态: {status}")
        if duration:
            print(f"   耗时: {duration:.2f}秒")
        
        joints_before = action.get('joint_angles_before')
        joints_after = action.get('joint_angles_after')
        if joints_before or joints_after:
            print(f"   关节角度变化:")
            for j in range(6):
                before = f"{joints_before[j]:.2f}°" if joints_before and len(joints_before) > j else "N/A"
                after = f"{joints_after[j]:.2f}°" if joints_after and len(joints_after) > j else "N/A"
                print(f"      关节{j+1}: {before} -> {after}")
        
        coords_before = action.get('coords_before')
        coords_after = action.get('coords_after')
        if coords_before or coords_after:
            print(f"   末端坐标变化:")
            coord_names = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
            for c in range(6):
                before = f"{coords_before[c]:.2f}" if coords_before and len(coords_before) > c else "N/A"
                after = f"{coords_after[c]:.2f}" if coords_after and len(coords_after) > c else "N/A"
                unit = "mm" if c < 3 else "°"
                print(f"      {coord_names[c]}: {before} {unit} -> {after} {unit}")
        
        if error:
            print(f"   错误: {error}")
    
    print()
    print("=" * 60)
    print("回放完成")
    print("=" * 60)

if __name__ == '__main__':
    playback_summary()
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f'[Visualizer] 回放脚本已生成: {output_path}')
    return output_path

def create_html_report(task_dir: str, output_path: Optional[str] = None) -> str:
    '''
    创建HTML交互式报告
    
    Args:
        task_dir: 任务目录
        output_path: 输出HTML路径
    
    Returns:
        HTML路径
    '''
    record = load_task_record(task_dir)
    task_id = record.get('task_id', 'unknown')
    
    action_executions = record.get('action_executions', [])
    vlm_results = record.get('vlm_results', [])
    camera_frames = record.get('camera_frames', [])
    speech_records = record.get('speech_records', [])
    
    status_colors = {
        'success': '#28a745',
        'failed': '#dc3545',
        'running': '#007bff',
        'pending': '#6c757d',
        'cancelled': '#ffc107'
    }
    
    status_badges = {
        'success': 'badge-success',
        'failed': 'badge-danger',
        'running': 'badge-primary',
        'pending': 'badge-secondary',
        'cancelled': 'badge-warning'
    }
    
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>任务复盘报告 - {task_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .meta {{ opacity: 0.9; font-size: 14px; }}
        .card {{ background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .card h2 {{ color: #667eea; font-size: 20px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e9ecef; }}
        .card h3 {{ color: #495057; font-size: 16px; margin: 15px 0 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e9ecef; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #495057; }}
        tr:hover {{ background: #f8f9fa; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-primary {{ background: #cce5ff; color: #004085; }}
        .badge-secondary {{ background: #e2e3e5; color: #383d41; }}
        .timeline {{ position: relative; padding-left: 30px; }}
        .timeline::before {{ content: ''; position: absolute; left: 10px; top: 0; bottom: 0; width: 2px; background: #e9ecef; }}
        .timeline-item {{ position: relative; margin-bottom: 25px; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .timeline-item::before {{ content: ''; position: absolute; left: -25px; top: 20px; width: 12px; height: 12px; border-radius: 50%; background: #667eea; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.15); }}
        .timeline-item.success::before {{ background: #28a745; }}
        .timeline-item.failed::before {{ background: #dc3545; }}
        .timeline-item.cancelled::before {{ background: #ffc107; }}
        .timeline-title {{ font-weight: 600; margin-bottom: 8px; }}
        .timeline-meta {{ font-size: 12px; color: #6c757d; margin-bottom: 10px; }}
        .error-box {{ background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #dc3545; }}
        .images-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin-top: 15px; }}
        .image-card {{ border: 1px solid #e9ecef; border-radius: 8px; overflow: hidden; }}
        .image-card img {{ width: 100%; height: auto; display: block; }}
        .image-card .caption {{ padding: 10px; background: #f8f9fa; font-size: 13px; color: #495057; }}
        pre {{ background: #282c34; color: #abb2bf; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; }}
        code {{ font-family: 'Fira Code', 'Consolas', monospace; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-card .value {{ font-size: 32px; font-weight: 700; }}
        .stat-card .label {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
        .stat-card.success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .stat-card.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .speech-bubble {{ background: #e8f0fe; padding: 15px; border-radius: 15px; position: relative; margin: 10px 0; }}
        .speech-bubble::before {{ content: ''; position: absolute; left: 20px; top: -10px; border-width: 0 10px 10px; border-style: solid; border-color: transparent transparent #e8f0fe; }}
        .footer {{ text-align: center; padding: 20px; color: #6c757d; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 任务复盘报告</h1>
            <div class="meta">
                任务ID: {task_id} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{len(action_executions)}</div>
                <div class="label">总动作数</div>
            </div>
            <div class="stat-card success">
                <div class="value">{sum(1 for a in action_executions if a.get('status') == 'success')}</div>
                <div class="label">成功</div>
            </div>
            <div class="stat-card warning">
                <div class="value">{sum(1 for a in action_executions if a.get('status') == 'failed')}</div>
                <div class="label">失败</div>
            </div>
            <div class="stat-card">
                <div class="value">{len(vlm_results)}</div>
                <div class="label">VLM调用</div>
            </div>
        </div>
'''
    
    if record.get('error_summary'):
        html_content += f'''
        <div class="card">
            <h2>⚠️ 错误摘要</h2>
            <div class="error-box">
                {record['error_summary']}
            </div>
        </div>
'''
    
    if speech_records:
        html_content += '''
        <div class="card">
            <h2>🎤 语音记录</h2>
'''
        for idx, speech in enumerate(speech_records, 1):
            text = speech.get('text', 'N/A')
            duration = speech.get('duration', 'N/A')
            html_content += f'''
            <h3>第 {idx} 条语音</h3>
            <div class="speech-bubble">
                <strong>语音文本:</strong> {text}<br>
                <strong>时长:</strong> {duration} 秒
            </div>
'''
        html_content += '''
        </div>
'''
    
    action_plan = record.get('action_plan')
    if action_plan:
        html_content += f'''
        <div class="card">
            <h2>📝 动作规划</h2>
            
            <h3>原始指令</h3>
            <div class="speech-bubble">{action_plan.get('original_instruction', 'N/A')}</div>
            
            <h3>智能体回复</h3>
            <div class="speech-bubble">{action_plan.get('agent_response', 'N/A')}</div>
            
            <h3>规划的动作序列</h3>
            <table>
                <tr><th>序号</th><th>动作</th></tr>
'''
        for idx, func in enumerate(action_plan.get('function_calls', []), 1):
            html_content += f'''                <tr><td>{idx}</td><td><code>{func}</code></td></tr>
'''
        html_content += '''            </table>
        </div>
'''
    
    if action_executions:
        html_content += '''
        <div class="card">
            <h2>⚙️ 动作执行时序</h2>
            <table>
                <tr><th>序号</th><th>动作</th><th>状态</th><th>耗时</th></tr>
'''
        for idx, action in enumerate(action_executions, 1):
            status = action.get('status', 'unknown')
            badge_class = status_badges.get(status, 'badge-secondary')
            duration = f"{action.get('duration'):.2f}s" if action.get('duration') else 'N/A'
            html_content += f'''                <tr><td>{idx}</td><td><code>{action.get('function_name')}</code></td><td><span class="badge {badge_class}">{status}</span></td><td>{duration}</td></tr>
'''
        html_content += '''            </table>
        </div>
        
        <div class="card">
            <h2>📋 详细执行记录</h2>
            <div class="timeline">
'''
        for idx, action in enumerate(action_executions, 1):
            status = action.get('status', 'unknown')
            func_name = action.get('function_name', 'unknown')
            duration = f"{action.get('duration'):.2f}秒" if action.get('duration') else 'N/A'
            badge_class = status_badges.get(status, 'badge-secondary')
            
            html_content += f'''
                <div class="timeline-item {status}">
                    <div class="timeline-title">
                        动作 {idx}: <code>{func_name}</code>
                        <span class="badge {badge_class}">{status}</span>
                    </div>
                    <div class="timeline-meta">耗时: {duration}</div>
'''
            
            joints_before = action.get('joint_angles_before')
            joints_after = action.get('joint_angles_after')
            if joints_before or joints_after:
                html_content += '''
                    <h4>关节角度变化</h4>
                    <table>
                        <tr><th>关节</th><th>执行前</th><th>执行后</th></tr>
'''
                for j in range(6):
                    before = f"{joints_before[j]:.2f}°" if joints_before and len(joints_before) > j else 'N/A'
                    after = f"{joints_after[j]:.2f}°" if joints_after and len(joints_after) > j else 'N/A'
                    html_content += f'''                        <tr><td>关节{j+1}</td><td>{before}</td><td>{after}</td></tr>
'''
                html_content += '''                    </table>
'''
            
            coords_before = action.get('coords_before')
            coords_after = action.get('coords_after')
            if coords_before or coords_after:
                html_content += '''
                    <h4>末端坐标变化</h4>
                    <table>
                        <tr><th>坐标</th><th>执行前</th><th>执行后</th></tr>
'''
                coord_names = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
                for c in range(6):
                    unit = 'mm' if c < 3 else '°'
                    before = f"{coords_before[c]:.2f} {unit}" if coords_before and len(coords_before) > c else 'N/A'
                    after = f"{coords_after[c]:.2f} {unit}" if coords_after and len(coords_after) > c else 'N/A'
                    html_content += f'''                        <tr><td>{coord_names[c]}</td><td>{before}</td><td>{after}</td></tr>
'''
                html_content += '''                    </table>
'''
            
            error_msg = action.get('error_message')
            if error_msg:
                html_content += f'''
                    <div class="error-box">
                        <strong>错误:</strong> {error_msg}
                    </div>
'''
            
            ret_val = action.get('return_value')
            if ret_val:
                html_content += f'''
                    <h4>返回值</h4>
                    <pre><code>{ret_val}</code></pre>
'''
            
            html_content += '''
                </div>
'''
        
        html_content += '''
            </div>
        </div>
'''
    
    if vlm_results:
        html_content += '''
        <div class="card">
            <h2>👁️ VLM识别结果</h2>
'''
        for idx, vlm in enumerate(vlm_results, 1):
            task_type = vlm.get('task_type', 'N/A')
            prompt = vlm.get('prompt', 'N/A')
            model = vlm.get('model_used', 'N/A')
            
            html_content += f'''
            <h3>第 {idx} 次VLM调用</h3>
            <table>
                <tr><th>任务类型</th><td>{task_type}</td></tr>
                <tr><th>使用模型</th><td><code>{model}</code></td></tr>
                <tr><th>提示词</th><td>{prompt}</td></tr>
            </table>
            
            <h4>解析结果</h4>
            <pre><code>{json.dumps(vlm.get('parsed_result', {}), ensure_ascii=False, indent=2)}</code></pre>
'''
            
            img_path = vlm.get('image_path', '')
            if img_path and os.path.exists(os.path.join(task_dir, img_path)):
                html_content += f'''
            <h4>输入图像</h4>
            <img src="{img_path}" alt="输入图像" style="max-width: 100%; border-radius: 8px; margin: 10px 0;">
'''
            
            viz_path = vlm.get('viz_path', '')
            if viz_path and os.path.exists(os.path.join(task_dir, viz_path)):
                html_content += f'''
            <h4>可视化结果</h4>
            <img src="{viz_path}" alt="可视化结果" style="max-width: 100%; border-radius: 8px; margin: 10px 0;">
'''
        
        html_content += '''
        </div>
'''
    
    if camera_frames:
        html_content += '''
        <div class="card">
            <h2>📷 相机关键帧</h2>
            <div class="images-grid">
'''
        for idx, frame in enumerate(camera_frames, 1):
            frame_path = frame.get('frame_path', '')
            description = frame.get('description', f'帧 {idx}')
            
            if frame_path and os.path.exists(os.path.join(task_dir, frame_path)):
                html_content += f'''
                <div class="image-card">
                    <img src="{frame_path}" alt="{description}">
                    <div class="caption">
                        <strong>{description}</strong><br>
                        <small>帧 {idx}</small>
                    </div>
                </div>
'''
        
        html_content += '''
            </div>
        </div>
'''
    
    html_content += '''
        <div class="footer">
            <p>机械臂智能体任务复盘系统 | 同济子豪兄</p>
        </div>
    </div>
</body>
</html>
'''
    
    if output_path is None:
        output_path = os.path.join(task_dir, 'report.html')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f'[Visualizer] HTML报告已生成: {output_path}')
    return output_path
