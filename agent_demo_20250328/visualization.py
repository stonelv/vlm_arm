# visualization.py
# 轨迹和状态可视化模块
# 提供轨迹图、状态图等可视化功能

import os
import json
import numpy as np
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[警告] matplotlib未安装，可视化图形功能不可用")
    print("        可运行: pip install matplotlib 安装")


class RobotVisualizer:
    '''
    机械臂轨迹和状态可视化器
    支持：
    - 3D轨迹图
    - 关节角度变化图
    - 吸泵状态时间线
    - 动作序列图
    '''
    
    def __init__(self, output_dir='visualizations'):
        '''
        初始化可视化器
        :param output_dir: 输出目录
        '''
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 颜色配置
        self.colors = {
            'trajectory': '#1f77b4',
            'start_point': '#2ca02c',
            'end_point': '#ff7f0e',
            'joint1': '#1f77b4',
            'joint2': '#ff7f0e',
            'joint3': '#2ca02c',
            'joint4': '#d62728',
            'joint5': '#9467bd',
            'joint6': '#8c564b',
            'pump_on': '#2ca02c',
            'pump_off': '#d62728',
        }
    
    def plot_3d_trajectory(self, trajectory, title='机械臂轨迹图', save_as=None):
        '''
        绘制3D轨迹图
        :param trajectory: 轨迹数据列表
        :param title: 图表标题
        :param save_as: 保存文件名（不含路径）
        '''
        if not MATPLOTLIB_AVAILABLE:
            print("[错误] matplotlib未安装，无法绘制3D轨迹图")
            return None
        
        # 提取坐标点
        coords_list = []
        for action in trajectory:
            state_before = action.get('state_before', {})
            coords = state_before.get('coords', [])
            if len(coords) >= 3:
                coords_list.append(coords[:3])  # 只取x, y, z
        
        if not coords_list:
            print("[警告] 轨迹中没有坐标数据")
            return None
        
        coords_array = np.array(coords_list)
        x = coords_array[:, 0]
        y = coords_array[:, 1]
        z = coords_array[:, 2]
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹线
        ax.plot(x, y, z, color=self.colors['trajectory'], linewidth=2, label='轨迹')
        
        # 绘制起点和终点
        ax.scatter(x[0], y[0], z[0], color=self.colors['start_point'], 
                   s=100, marker='o', label='起点')
        ax.scatter(x[-1], y[-1], z[-1], color=self.colors['end_point'], 
                   s=100, marker='*', label='终点')
        
        # 设置标签和标题
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        ax.set_zlabel('Z 高度', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 保存或显示
        if save_as:
            save_path = os.path.join(self.output_dir, save_as)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[可视化] 3D轨迹图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_joint_angles(self, trajectory, title='关节角度变化图', save_as=None):
        '''
        绘制关节角度变化图
        :param trajectory: 轨迹数据列表
        :param title: 图表标题
        :param save_as: 保存文件名
        '''
        if not MATPLOTLIB_AVAILABLE:
            print("[错误] matplotlib未安装，无法绘制关节角度图")
            return None
        
        # 提取关节角度
        angles_list = []
        for action in trajectory:
            state_before = action.get('state_before', {})
            angles = state_before.get('angles', [])
            if len(angles) == 6:
                angles_list.append(angles)
        
        if not angles_list:
            print("[警告] 轨迹中没有角度数据")
            return None
        
        angles_array = np.array(angles_list)
        time_steps = np.arange(len(angles_list))
        
        # 创建子图
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        joint_names = ['关节1 (底座旋转)', '关节2 (肩关节)', '关节3 (肘关节)', 
                       '关节4 (腕关节1)', '关节5 (腕关节2)', '关节6 (末端旋转)']
        
        for i, ax in enumerate(axes.flat):
            color = list(self.colors.values())[i + 3]  # 从joint1开始
            ax.plot(time_steps, angles_array[:, i], color=color, linewidth=2)
            ax.set_xlabel('动作序号', fontsize=10)
            ax.set_ylabel('角度 (度)', fontsize=10)
            ax.set_title(joint_names[i], fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-180, 180])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存或显示
        if save_as:
            save_path = os.path.join(self.output_dir, save_as)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[可视化] 关节角度图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_action_sequence(self, trajectory, title='动作序列图', save_as=None):
        '''
        绘制动作序列图
        :param trajectory: 轨迹数据列表
        :param title: 图表标题
        :param save_as: 保存文件名
        '''
        if not MATPLOTLIB_AVAILABLE:
            print("[错误] matplotlib未安装，无法绘制动作序列图")
            return None
        
        if not trajectory:
            print("[警告] 轨迹为空")
            return None
        
        # 统计各种动作类型
        action_types = []
        for action in trajectory:
            action_types.append(action.get('action_type', 'unknown'))
        
        unique_actions = list(set(action_types))
        action_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_actions)))
        color_map = dict(zip(unique_actions, action_colors))
        
        # 创建颜色列表
        colors = [color_map[at] for at in action_types]
        
        # 绘制
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 绘制时间线
        y_pos = np.ones(len(action_types))
        ax.bar(range(len(action_types)), y_pos, color=colors, edgecolor='white', linewidth=1)
        
        # 设置标签
        ax.set_xlabel('动作序号', fontsize=12)
        ax.set_ylabel('动作类型', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 设置y轴
        ax.set_yticks([])
        ax.set_ylim([0, 1.2])
        
        # 添加动作类型标注
        for i, (at, color) in enumerate(zip(action_types, colors)):
            ax.annotate(at, (i, 0.5), ha='center', va='center', 
                       fontsize=8, rotation=45,
                       bbox=dict(boxstyle='round,pad=0.2', 
                                fc='white', ec='gray', alpha=0.8))
        
        # 创建图例
        legend_patches = [mpatches.Patch(color=color_map[at], label=at) 
                          for at in unique_actions]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
        
        # 设置x轴范围
        ax.set_xlim([-0.5, len(action_types) - 0.5])
        ax.set_xticks(range(0, len(action_types), max(1, len(action_types) // 10)))
        
        plt.tight_layout()
        
        # 保存或显示
        if save_as:
            save_path = os.path.join(self.output_dir, save_as)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[可视化] 动作序列图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def generate_trajectory_report(self, trajectory, report_name=None):
        '''
        生成完整的轨迹可视化报告
        :param trajectory: 轨迹数据
        :param report_name: 报告名称
        '''
        if report_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f'report_{timestamp}'
        
        print(f"\n[可视化报告] 开始生成报告: {report_name}")
        print("=" * 50)
        
        # 统计信息
        total_actions = len(trajectory)
        action_types = {}
        for action in trajectory:
            at = action.get('action_type', 'unknown')
            action_types[at] = action_types.get(at, 0) + 1
        
        print(f"\n轨迹统计:")
        print(f"  总动作数: {total_actions}")
        print(f"  动作类型统计:")
        for at, count in action_types.items():
            print(f"    {at}: {count}")
        
        # 生成各种图表
        saved_files = []
        
        # 3D轨迹图
        if total_actions > 0:
            f = self.plot_3d_trajectory(
                trajectory, 
                title=f'轨迹图 - {report_name}',
                save_as=f'{report_name}_trajectory_3d.png'
            )
            if f:
                saved_files.append(f)
        
        # 关节角度图
        f = self.plot_joint_angles(
            trajectory,
            title=f'关节角度变化 - {report_name}',
            save_as=f'{report_name}_joint_angles.png'
        )
        if f:
            saved_files.append(f)
        
        # 动作序列图
        f = self.plot_action_sequence(
            trajectory,
            title=f'动作序列 - {report_name}',
            save_as=f'{report_name}_action_sequence.png'
        )
        if f:
            saved_files.append(f)
        
        print(f"\n[可视化报告] 报告生成完成!")
        print(f"  生成的文件:")
        for f in saved_files:
            print(f"    - {f}")
        
        return saved_files


# 便捷函数
def visualize_trajectory_file(json_file, report_name=None):
    '''
    从JSON文件加载轨迹并生成可视化报告
    :param json_file: 轨迹JSON文件路径
    :param report_name: 报告名称
    '''
    print(f"\n[可视化] 加载轨迹文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trajectory = data.get('trajectory', [])
    
    if not trajectory:
        print("[错误] 文件中没有轨迹数据")
        return None
    
    print(f"[可视化] 共加载 {len(trajectory)} 个动作")
    
    visualizer = RobotVisualizer()
    return visualizer.generate_trajectory_report(trajectory, report_name)


def quick_visualize(trajectory, output_prefix='quick_viz'):
    '''
    快速可视化（无需创建实例）
    :param trajectory: 轨迹数据
    :param output_prefix: 输出文件名前缀
    '''
    visualizer = RobotVisualizer()
    return visualizer.generate_trajectory_report(trajectory, output_prefix)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了参数，假设是轨迹文件路径
        json_file = sys.argv[1]
        report_name = sys.argv[2] if len(sys.argv) > 2 else None
        visualize_trajectory_file(json_file, report_name)
    else:
        print("用法: python visualization.py <轨迹文件.json> [报告名称]")
        print("示例: python visualization.py temp/trajectory_log.json my_report")
