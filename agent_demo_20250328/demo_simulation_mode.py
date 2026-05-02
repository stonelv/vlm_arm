# demo_simulation_mode.py
# 仿真模式使用示例
# 展示如何在无硬件环境下开发和测试机械臂控制代码

import os
import sys

print("=" * 60)
print("仿真模式使用示例")
print("=" * 60)

# 方式1: 通过环境变量设置模式（推荐）
# 在命令行运行: ROBOT_MODE=simulation python demo_simulation_mode.py

# 方式2: 代码中动态设置模式（仅用于演示，实际推荐用环境变量）
# from config import set_robot_mode
# set_robot_mode('simulation')  # 设置为仿真模式
# set_robot_mode('real')         # 设置为真机模式

from config import is_simulation_mode, get_robot_mode

print(f"\n当前运行模式: {get_robot_mode()}")
print(f"是否为仿真模式: {is_simulation_mode()}")

# 导入机器人模块（会自动根据模式选择实现）
print("\n导入机器人模块...")
from utils_robot import (
    mc, back_zero, move_to_coords, single_joint_move,
    save_trajectory, load_trajectory, playback_trajectory,
    get_robot_instance
)
from utils_pump import pump_on, pump_off

def demo_basic_operations():
    '''演示基本操作'''
    print("\n" + "-" * 40)
    print("演示: 基本操作")
    print("-" * 40)
    
    # 归零
    print("\n1. 机械臂归零")
    back_zero()
    
    # 移动到指定坐标
    print("\n2. 移动到指定坐标")
    move_to_coords(X=150, Y=-100, HEIGHT_SAFE=200)
    
    # 单关节移动
    print("\n3. 单关节移动")
    single_joint_move(1, 30)  # 关节1移动到30度
    single_joint_move(1, 0)   # 关节1回到0度

def demo_pump_control():
    '''演示吸泵控制'''
    print("\n" + "-" * 40)
    print("演示: 吸泵控制")
    print("-" * 40)
    
    print("\n1. 开启吸泵")
    pump_on()
    
    # 模拟等待
    import time
    time.sleep(0.1)
    
    print("\n2. 关闭吸泵")
    pump_off()

def demo_trajectory_features():
    '''演示轨迹记录和回放（仅仿真模式）'''
    if not is_simulation_mode():
        print("\n" + "-" * 40)
        print("轨迹功能演示（仅仿真模式可用）")
        print("-" * 40)
        print("当前为真机模式，跳过轨迹功能演示")
        return
    
    print("\n" + "-" * 40)
    print("演示: 轨迹记录和回放（仿真模式专用）")
    print("-" * 40)
    
    # 获取机器人实例，访问仿真模式特有功能
    robot = get_robot_instance()
    
    # 执行一些动作，这些动作会被自动记录
    print("\n1. 执行动作并自动记录轨迹...")
    robot.send_angles([0, 0, 0, 0, 0, 0], 40)
    robot.send_coords([150, -100, 200, 0, 180, 90], 20, 0)
    robot.send_angle(1, 30, 50)
    robot.send_angle(1, 0, 50)
    robot.send_coords([100, -50, 150, 0, 180, 90], 20, 0)
    
    # 查看当前状态
    state = robot.get_state_summary()
    print(f"\n   已记录 {state['trajectory_length']} 个动作")
    print(f"   当前角度: {state['current_angles']}")
    print(f"   当前坐标: {state['current_coords']}")
    
    # 保存轨迹到文件
    print("\n2. 保存轨迹到文件...")
    save_trajectory('temp/demo_trajectory.json')
    print("   轨迹已保存到: temp/demo_trajectory.json")
    
    # 加载轨迹
    print("\n3. 从文件加载轨迹...")
    trajectory = load_trajectory('temp/demo_trajectory.json')
    print(f"   成功加载 {len(trajectory)} 个动作")
    
    # 回放轨迹
    print("\n4. 回放加载的轨迹...")
    print("   创建新的机器人实例用于回放...")
    
    from simulated_robot import SimulatedRobot
    playback_robot = SimulatedRobot(enable_visualization=True)
    
    print("\n   开始回放（加速模式）...")
    playback_robot.playback_trajectory(trajectory, speed_multiplier=0.5)
    
    print("\n   回放完成!")

def demo_visualization():
    '''演示可视化输出（仅仿真模式）'''
    if not is_simulation_mode():
        print("\n" + "-" * 40)
        print("可视化演示（仅仿真模式可用）")
        print("-" * 40)
        print("当前为真机模式，跳过可视化演示")
        return
    
    print("\n" + "-" * 40)
    print("演示: 可视化输出（仿真模式专用）")
    print("-" * 40)
    
    # 创建带可视化的机器人实例
    from simulated_robot import SimulatedRobot
    viz_robot = SimulatedRobot(enable_visualization=True)
    
    print("\n执行动作，观察控制台可视化输出:")
    print("-" * 50)
    
    # 执行一系列动作
    viz_robot.send_angles([0, 0, 0, 0, 0, 0], 40)
    viz_robot.send_coords([200, -50, 180, 0, 180, 90], 20, 0)
    viz_robot.send_angle(2, -45, 50)
    viz_robot.set_fresh_mode(0)
    viz_robot.send_coords([150, -100, 200, 0, 180, 90], 20, 0)

def main():
    '''主函数'''
    print("\n" + "=" * 60)
    print("开始演示")
    print("=" * 60)
    
    # 演示基本操作
    demo_basic_operations()
    
    # 演示吸泵控制
    demo_pump_control()
    
    # 演示轨迹功能（仅仿真模式）
    demo_trajectory_features()
    
    # 演示可视化（仅仿真模式）
    demo_visualization()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    # 如果是仿真模式，显示如何切换模式
    if is_simulation_mode():
        print("\n💡 提示:")
        print("   - 当前为仿真模式，无需真实硬件即可测试")
        print("   - 要切换到真机模式，设置环境变量: ROBOT_MODE=real")
        print("   - 两种模式接口完全兼容，代码无需修改")
        print("   - 仿真模式下可以记录和回放轨迹，用于调试和分析")
    else:
        print("\n💡 提示:")
        print("   - 当前为真机模式，连接真实硬件")
        print("   - 要切换到仿真模式测试，设置环境变量: ROBOT_MODE=simulation")
        print("   - 仿真模式可用于无硬件环境下的开发和测试")

if __name__ == '__main__':
    # 确保 temp 目录存在
    os.makedirs('temp', exist_ok=True)
    
    main()
