# utils_review.py
# 同济子豪兄 2024-5-27
# 一键复盘接口：整合报告生成与可视化回放

import os
import sys
from typing import Optional, Dict, Any

RECORD_DIR = 'task_records'

def review_task(task_dir: str, generate_all: bool = True, 
                show_plots: bool = False) -> Dict[str, Any]:
    '''
    一键复盘指定任务
    
    Args:
        task_dir: 任务目录路径
        generate_all: 是否生成所有报告和可视化
        show_plots: 是否显示图表（仅用于调试）
    
    Returns:
        包含生成文件路径的字典
    '''
    from utils_report import generate_markdown_report, list_all_tasks
    from utils_visualizer import (
        generate_all_charts, 
        create_playback_script,
        create_html_report
    )
    
    if not os.path.exists(task_dir):
        raise FileNotFoundError(f'任务目录不存在: {task_dir}')
    
    print(f"\n{'='*60}")
    print(f"开始复盘任务: {os.path.basename(task_dir)}")
    print(f"{'='*60}\n")
    
    results = {
        'task_dir': task_dir,
        'files': {}
    }
    
    print("[1/4] 生成Markdown报告...")
    try:
        md_path = generate_markdown_report(task_dir)
        results['files']['markdown'] = md_path
        print(f"    ✓ Markdown报告: {md_path}")
    except Exception as e:
        print(f"    ✗ 生成Markdown报告失败: {e}")
    
    print("\n[2/4] 生成可视化图表...")
    try:
        charts = generate_all_charts(task_dir, show_plot=show_plots)
        results['files']['charts'] = charts
        for chart_type, path in charts.items():
            print(f"    ✓ {chart_type}图表: {path}")
    except Exception as e:
        print(f"    ✗ 生成图表失败: {e}")
    
    print("\n[3/4] 创建回放脚本...")
    try:
        script_path = create_playback_script(task_dir)
        results['files']['playback_script'] = script_path
        print(f"    ✓ 回放脚本: {script_path}")
    except Exception as e:
        print(f"    ✗ 创建回放脚本失败: {e}")
    
    print("\n[4/4] 生成HTML交互式报告...")
    try:
        html_path = create_html_report(task_dir)
        results['files']['html'] = html_path
        print(f"    ✓ HTML报告: {html_path}")
    except Exception as e:
        print(f"    ✗ 生成HTML报告失败: {e}")
    
    print(f"\n{'='*60}")
    print("复盘完成!")
    print(f"{'='*60}\n")
    
    return results

def review_latest_task() -> Optional[Dict[str, Any]]:
    '''
    一键复盘最近一次任务
    
    Returns:
        复盘结果，如果没有任务记录则返回None
    '''
    from utils_report import list_all_tasks
    
    tasks = list_all_tasks()
    if not tasks:
        print("[Review] 没有找到任何任务记录")
        return None
    
    latest_task = tasks[0]
    print(f"[Review] 复盘最近任务: {latest_task['task_id']}")
    print(f"[Review] 任务状态: {latest_task['status']}")
    print(f"[Review] 任务时长: {latest_task['total_duration']}")
    
    return review_task(latest_task['task_dir'])

def review_all_tasks() -> list:
    '''
    复盘所有任务
    
    Returns:
        所有任务的复盘结果列表
    '''
    from utils_report import list_all_tasks
    
    tasks = list_all_tasks()
    if not tasks:
        print("[Review] 没有找到任何任务记录")
        return []
    
    results = []
    for task in tasks:
        print(f"\n[Review] 复盘任务: {task['task_id']}")
        try:
            result = review_task(task['task_dir'])
            results.append(result)
        except Exception as e:
            print(f"[Review] 复盘任务失败 {task['task_id']}: {e}")
            results.append({'task_id': task['task_id'], 'error': str(e)})
    
    return results

def list_tasks() -> None:
    '''
    列出所有任务记录
    '''
    from utils_report import list_all_tasks
    
    tasks = list_all_tasks()
    
    if not tasks:
        print("[Review] 没有找到任何任务记录")
        return
    
    print(f"\n{'='*80}")
    print(f"{'任务列表':^80}")
    print(f"{'='*80}")
    print(f"{'任务ID':<22} {'状态':<10} {'开始时间':<25} {'时长':<10}")
    print(f"{'-'*80}")
    
    for task in tasks:
        status_icon = {
            'success': '✅',
            'failed': '❌',
            'running': '⏳',
            'pending': '📋',
            'cancelled': '🚫'
        }.get(task.get('status', 'unknown'), '❓')
        
        print(
            f"{task['task_id']:<22} "
            f"{status_icon} {task['status']:<8} "
            f"{task['start_time']:<25} "
            f"{task['total_duration']:<10}"
        )
        
        if task.get('error_summary'):
            error_preview = task['error_summary'][:50]
            print(f"{'':<22}   Error: {error_preview}...")
    
    print(f"{'='*80}")
    print(f"共 {len(tasks)} 个任务记录\n")

def show_task_summary(task_dir: str) -> None:
    '''
    显示任务摘要（不生成报告）
    
    Args:
        task_dir: 任务目录
    '''
    import json
    
    json_path = os.path.join(task_dir, 'task_record.json')
    if not os.path.exists(json_path):
        print(f"[Review] 任务记录不存在: {task_dir}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"任务摘要 - {record.get('task_id', 'unknown')}")
    print(f"{'='*60}")
    
    print(f"\n【基本信息】")
    print(f"  开始时间: {record.get('start_time', 'N/A')}")
    print(f"  结束时间: {record.get('end_time', 'N/A')}")
    print(f"  总耗时: {record.get('total_duration', 'N/A')} 秒")
    print(f"  状态: {record.get('status', 'N/A')}")
    
    if record.get('error_summary'):
        print(f"\n【错误摘要】")
        print(f"  {record['error_summary']}")
    
    speech_records = record.get('speech_records', [])
    if speech_records:
        print(f"\n【语音记录】({len(speech_records)}条)")
        for i, s in enumerate(speech_records, 1):
            text_preview = s.get('text', '')[:50]
            print(f"  {i}. {text_preview}...")
    
    action_plan = record.get('action_plan')
    if action_plan:
        print(f"\n【动作规划】")
        print(f"  指令: {action_plan.get('original_instruction', 'N/A')}")
        print(f"  回复: {action_plan.get('agent_response', 'N/A')}")
        
        funcs = action_plan.get('function_calls', [])
        if funcs:
            print(f"  规划动作 ({len(funcs)}个):")
            for i, f in enumerate(funcs, 1):
                print(f"    {i}. {f}")
    
    action_executions = record.get('action_executions', [])
    if action_executions:
        print(f"\n【执行统计】")
        success = sum(1 for a in action_executions if a.get('status') == 'success')
        failed = sum(1 for a in action_executions if a.get('status') == 'failed')
        print(f"  总动作: {len(action_executions)}")
        print(f"  成功: {success}")
        print(f"  失败: {failed}")
    
    vlm_results = record.get('vlm_results', [])
    camera_frames = record.get('camera_frames', [])
    print(f"\n【视觉记录】")
    print(f"  VLM调用: {len(vlm_results)} 次")
    print(f"  相机帧: {len(camera_frames)} 帧")
    
    print(f"\n{'='*60}\n")

def main():
    '''
    命令行入口
    '''
    import argparse
    
    parser = argparse.ArgumentParser(description='机械臂智能体任务复盘工具')
    parser.add_argument('-l', '--list', action='store_true', help='列出所有任务记录')
    parser.add_argument('-s', '--summary', type=str, help='显示指定任务的摘要（任务目录路径或任务ID）')
    parser.add_argument('-r', '--review', type=str, help='复盘指定任务（任务目录路径或任务ID）')
    parser.add_argument('--latest', action='store_true', help='复盘最近一次任务')
    parser.add_argument('--all', action='store_true', help='复盘所有任务')
    parser.add_argument('--show-plots', action='store_true', help='显示图表（调试用）')
    
    args = parser.parse_args()
    
    if args.list:
        list_tasks()
        return
    
    if args.summary:
        task_path = args.summary
        if not os.path.isdir(task_path):
            task_path = os.path.join(RECORD_DIR, task_path)
        show_task_summary(task_path)
        return
    
    if args.latest:
        review_latest_task()
        return
    
    if args.all:
        review_all_tasks()
        return
    
    if args.review:
        task_path = args.review
        if not os.path.isdir(task_path):
            task_path = os.path.join(RECORD_DIR, task_path)
        review_task(task_path, show_plots=args.show_plots)
        return
    
    parser.print_help()
    print(f"""
示例用法:
  python utils_review.py -l                    # 列出所有任务
  python utils_review.py -s 20240527_143000   # 显示任务摘要
  python utils_review.py -r 20240527_143000   # 复盘指定任务
  python utils_review.py --latest               # 复盘最近任务
  python utils_review.py --all                  # 复盘所有任务
""")

if __name__ == '__main__':
    main()
