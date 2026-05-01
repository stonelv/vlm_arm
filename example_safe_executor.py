#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全动作执行器 - 最小可运行示例
====================================

本示例演示如何使用 SafeActionExecutor 进行安全的机械臂控制。

运行方式：
    python example_safe_executor.py
"""

import time
import json

from safe_action_executor import (
    SafeActionExecutor,
    Pose,
    JointAngles,
    HighLevelCommand,
    ActionType,
    SafetyConstraints,
    create_executor,
    SafetyException,
    WorkspaceLimitExceeded,
    JointLimitExceeded,
    CollisionDetected,
)


def example_1_basic_move():
    """
    示例1: 基础移动 - 移动到指定位姿
    """
    print("\n" + "="*70)
    print("示例1: 基础移动 - 移动到指定位姿")
    print("="*70)
    
    # 创建执行器（仿真模式）
    executor = create_executor(simulation=True)
    
    try:
        # 显示初始状态
        state = executor.get_state()
        print(f"初始状态: {json.dumps(state, indent=2, ensure_ascii=False)}")
        
        # 方式1: 使用字典执行指令
        print("\n[方式1] 使用字典执行移动指令...")
        command_dict = {
            'action': 'move',
            'pose': [200, -100, 220, 0, 180, 90],  # [x, y, z, rx, ry, rz]
            'speed': 30
        }
        executor.execute_from_dict(command_dict)
        
        # 方式2: 使用 HighLevelCommand 对象
        print("\n[方式2] 使用 HighLevelCommand 对象执行移动...")
        target_pose = Pose(x=150, y=100, z=200, rx=0, ry=180, rz=90)
        command = HighLevelCommand(
            action_type=ActionType.MOVE_TO_POSE,
            target_pose=target_pose,
            speed=40
        )
        executor.execute_command(command)
        
        # 方式3: 移动关节
        print("\n[方式3] 移动到指定关节角度...")
        joint_command = {
            'action': 'move_joints',
            'joints': [30.0, -20.0, 45.0, 0.0, 10.0, -15.0],  # 6个关节角度
            'speed': 25
        }
        executor.execute_from_dict(joint_command)
        
        # 归零
        print("\n[归零] 回到零位...")
        executor.execute_from_dict({'action': 'home', 'speed': 30})
        
        print("\n示例1完成!")
        
    finally:
        executor.shutdown()


def example_2_pick_and_place():
    """
    示例2: 抓取和放置动作
    演示完整的 pick -> place 流程
    """
    print("\n" + "="*70)
    print("示例2: 抓取和放置动作 (Pick & Place)")
    print("="*70)
    
    executor = create_executor(simulation=True)
    
    try:
        # 抓取位置和放置位置
        pick_pose = Pose(x=200, y=-50, z=90, rx=0, ry=180, rz=90)
        place_pose = Pose(x=100, y=150, z=100, rx=0, ry=180, rz=90)
        
        print(f"抓取位置: {pick_pose}")
        print(f"放置位置: {place_pose}")
        
        # 步骤1: 执行抓取
        print("\n[步骤1] 执行抓取动作...")
        pick_command = {
            'action': 'pick',
            'pose': [pick_pose.x, pick_pose.y, pick_pose.z, 
                     pick_pose.rx, pick_pose.ry, pick_pose.rz],
            'height': 90.0,        # 抓取高度
            'safe_height': 220.0,  # 安全移动高度
            'speed': 30
        }
        executor.execute_from_dict(pick_command)
        
        # 步骤2: 执行放置
        print("\n[步骤2] 执行放置动作...")
        place_command = {
            'action': 'place',
            'pose': [place_pose.x, place_pose.y, place_pose.z,
                     place_pose.rx, place_pose.ry, place_pose.rz],
            'height': 100.0,
            'safe_height': 220.0,
            'speed': 30
        }
        executor.execute_from_dict(place_command)
        
        # 显示动作历史
        print("\n[动作历史]")
        for i, record in enumerate(executor.action_history.history):
            print(f"  [{i+1}] {record.action_type.value} - "
                  f"{'成功' if record.success else '失败'} - "
                  f"起始: {record.start_pose}")
        
        print("\n示例2完成!")
        
    finally:
        executor.shutdown()


def example_3_safety_constraints():
    """
    示例3: 安全约束演示
    演示限位、碰撞检测等安全功能
    """
    print("\n" + "="*70)
    print("示例3: 安全约束演示")
    print("="*70)
    
    # 使用自定义安全约束
    constraints = SafetyConstraints(
        max_speed=60,           # 最大速度60%
        min_speed=5,             # 最小速度5%
        default_safe_height=200,
        collision_distance_threshold=100,  # 碰撞预警距离100mm
        emergency_stop_distance=30,         # 紧急停止距离30mm
    )
    
    executor = SafeActionExecutor(simulation_mode=True, constraints=constraints)
    executor.initialize()
    
    try:
        # 场景1: 添加障碍物
        print("\n[场景1] 添加障碍物并测试碰撞检测")
        obstacle_pose = Pose(x=150, y=0, z=150)
        executor.add_obstacle(obstacle_pose, radius=60.0, name="桌子障碍物")
        
        # 尝试移动到障碍物附近（会触发警告）
        try:
            near_obstacle = Pose(x=150, y=50, z=150)
            print(f"尝试移动到障碍物附近: {near_obstacle}")
            executor.execute_from_dict({
                'action': 'move',
                'pose': near_obstacle.to_list(),
                'speed': 30
            })
        except Exception as e:
            print(f"检测到安全问题: {e}")
        
        # 场景2: 测试工作空间限制
        print("\n[场景2] 测试工作空间限制")
        try:
            # 尝试移动到工作空间之外
            invalid_pose = Pose(x=400, y=0, z=200)  # x超出限制
            print(f"尝试移动到无效位置: {invalid_pose}")
            executor.execute_from_dict({
                'action': 'move',
                'pose': invalid_pose.to_list(),
            })
        except WorkspaceLimitExceeded as e:
            print(f"工作空间限制触发: {e}")
        
        # 场景3: 测试关节限位
        print("\n[场景3] 测试关节限位")
        try:
            # 尝试设置超出限位的关节角度
            invalid_joints = [200.0, 0, 0, 0, 0, 0]  # 关节1超出165度限制
            print(f"尝试移动到无效关节角度: {invalid_joints}")
            executor.execute_from_dict({
                'action': 'move_joints',
                'joints': invalid_joints,
            })
        except JointLimitExceeded as e:
            print(f"关节限位触发: {e}")
        
        # 场景4: 速度约束
        print("\n[场景4] 测试速度约束")
        print(f"当前最大速度限制: {executor.constraints.max_speed}%")
        # 尝试使用超出限制的速度（会被自动约束）
        executor.execute_from_dict({
            'action': 'move',
            'pose': [100, 100, 200, 0, 180, 90],
            'speed': 90  # 超出max_speed=60，会被约束
        })
        
        print("\n示例3完成!")
        
    finally:
        executor.shutdown()


def example_4_llm_integration():
    """
    示例4: 大模型指令解析演示
    模拟从大模型响应中解析并执行指令
    """
    print("\n" + "="*70)
    print("示例4: 大模型指令解析演示")
    print("="*70)
    
    executor = create_executor(simulation=True)
    
    try:
        # 模拟大模型可能返回的各种响应格式
        
        # 格式1: 结构化JSON
        llm_response_1 = """
        分析完毕，我将执行以下操作：
        {
            "action": "move",
            "pose": [180, -80, 220, 0, 180, 90],
            "speed": 35
        }
        """
        print("\n[格式1] 从结构化JSON解析:")
        print(f"LLM响应: {llm_response_1.strip()}")
        executor.execute_from_llm(llm_response_1)
        
        # 格式2: 自然语言（简化关键词解析）
        llm_response_2 = "请移动到坐标 x=150, y=100, z=200 的位置，速度设置为40"
        print("\n[格式2] 从自然语言解析:")
        print(f"LLM响应: {llm_response_2}")
        executor.execute_from_llm(llm_response_2)
        
        # 格式3: 中文指令
        llm_response_3 = "抓取坐标 200, -50, 90 处的物体，高度为90"
        print("\n[格式3] 中文抓取指令:")
        print(f"LLM响应: {llm_response_3}")
        try:
            executor.execute_from_llm(llm_response_3)
        except Exception as e:
            print(f"执行提示: {e}")
        
        print("\n示例4完成!")
        
    finally:
        executor.shutdown()


def example_5_error_handling():
    """
    示例5: 错误处理和回滚机制演示
    """
    print("\n" + "="*70)
    print("示例5: 错误处理和回滚机制演示")
    print("="*70)
    
    executor = create_executor(simulation=True)
    
    try:
        print("\n[步骤1] 先执行几个成功的动作...")
        
        # 成功动作1
        executor.execute_from_dict({
            'action': 'move',
            'pose': [200, 0, 220, 0, 180, 90],
            'speed': 30
        })
        print(f"当前位置: {executor.robot.get_current_pose()}")
        
        # 成功动作2
        executor.execute_from_dict({
            'action': 'move',
            'pose': [150, 100, 200, 0, 180, 90],
            'speed': 30
        })
        print(f"当前位置: {executor.robot.get_current_pose()}")
        
        print("\n[步骤2] 查看动作历史...")
        last_safe = executor.action_history.get_last_successful_action()
        if last_safe:
            print(f"最近成功的动作起始位置: {last_safe.start_pose}")
        
        print("\n[步骤3] 模拟执行出错（手动触发）...")
        try:
            # 尝试执行无效操作
            executor.execute_from_dict({
                'action': 'move',
                'pose': [500, 0, 200, 0, 180, 90],  # x超出工作空间
            })
        except WorkspaceLimitExceeded as e:
            print(f"捕获到异常: {e}")
            print("异常已被安全处理，机械臂状态保持安全")
        
        # 检查当前状态仍然正常
        state = executor.get_state()
        print(f"\n最终状态:")
        print(f"  初始化: {state['initialized']}")
        print(f"  执行中: {state['is_executing']}")
        print(f"  当前位姿: {state['current_pose']}")
        
        print("\n示例5完成!")
        
    finally:
        executor.shutdown()


def example_6_real_vs_simulated():
    """
    示例6: 真机模式与仿真模式的统一接口演示
    """
    print("\n" + "="*70)
    print("示例6: 真机模式与仿真模式的统一接口")
    print("="*70)
    
    print("""
    安全动作执行器提供了统一的接口，代码在仿真模式和真机模式下无需修改。
    
    ==================================================================
    一、仿真模式 (用于测试和调试，无需连接真实机械臂)
    ==================================================================
    
    # 方式1: 使用 create_executor 便捷函数
    executor = create_executor(simulation=True)
    
    # 方式2: 直接使用 SafeActionExecutor
    executor = SafeActionExecutor(simulation_mode=True)
    executor.initialize()
    
    ==================================================================
    二、真机模式 (连接真实 MyCobot 机械臂)
    ==================================================================
    
    注意：真机模式需要安装 pymycobot 库:
        pip install pymycobot
    
    方式1: 使用默认端口（自动检测，适用于树莓派）
    executor = create_executor(simulation=False)
    
    方式2: 指定串口和波特率
    # Linux/Raspberry Pi: "/dev/ttyUSB0", "/dev/ttyAMA0", "/dev/ttyACM0"
    # Windows: "COM3", "COM5" 等
    # Mac: "/dev/tty.usbserial-1410" 等
    
    executor = SafeActionExecutor(
        simulation_mode=False,
        port="/dev/ttyUSB0",      # 串口设备路径
        baud=115200                # 波特率（可选，默认 115200）
    )
    executor.initialize()
    
    # 或使用便捷函数
    executor = create_executor(
        simulation=False,
        port="/dev/ttyUSB0",
        baud=115200
    )
    
    ==================================================================
    三、常用端口说明
    ==================================================================
    
    | 平台          | 常用端口                          |
    |---------------|-----------------------------------|
    | Raspberry Pi  | /dev/ttyAMA0 (默认), /dev/ttyUSB0|
    | Windows       | COM3, COM5 (查看设备管理器)      |
    | Mac           | /dev/tty.usbserial-*             |
    | Linux (PC)    | /dev/ttyUSB0, /dev/ttyACM0       |
    
    波特率: MyCobot 默认 115200
    
    ==================================================================
    四、统一的 API (两种模式下使用方式完全相同)
    ==================================================================
    
    以下代码在仿真和真机模式下无需修改:
        - executor.execute_from_dict(...)
        - executor.execute_command(...)
        - executor.execute_from_llm(...)
        - executor.add_obstacle(...)
        - executor.emergency_stop()
        - executor.shutdown()
    """)
    
    # 实际演示：相同代码在仿真模式下运行
    print("\n[实际演示] 相同代码在仿真模式下运行...")
    
    def common_robot_code(executor: SafeActionExecutor):
        """这段代码在仿真和真机模式下完全相同"""
        # 归零
        executor.execute_from_dict({'action': 'home', 'speed': 40})
        
        # 移动到工作位置
        executor.execute_from_dict({
            'action': 'move',
            'pose': [200, -100, 220, 0, 180, 90],
            'speed': 30
        })
        
        # 返回当前状态
        return executor.get_state()
    
    # 使用仿真模式执行
    executor = create_executor(simulation=True)
    try:
        result = common_robot_code(executor)
        print(f"执行完成，当前位姿: {result['current_pose']}")
    finally:
        executor.shutdown()
    
    print("\n提示: 只需修改 simulation_mode 参数即可切换到真机模式!")
    print("\n示例6完成!")


def main():
    """
    主函数 - 运行所有示例
    """
    print("="*70)
    print("安全动作执行器 - 完整示例演示")
    print("Safe Action Executor - Complete Demo")
    print("="*70)
    print(f"\n当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: 仿真模式 (所有示例均在仿真模式下运行)")
    
    # 运行所有示例
    try:
        example_1_basic_move()
        example_2_pick_and_place()
        example_3_safety_constraints()
        example_4_llm_integration()
        example_5_error_handling()
        example_6_real_vs_simulated()
        
        print("\n" + "="*70)
        print("所有示例执行完成!")
        print("="*70)
        print("""
        日志文件: safe_action_executor.log (已记录所有操作)
        
        核心功能总结:
        1. ✓ 高层指令接收 (move/pick/place/move_joints/home/relax)
        2. ✓ 安全约束 (关节限位、工作空间限制、速度约束)
        3. ✓ 碰撞检测 (障碍物距离检测)
        4. ✓ 异常回滚 (失败时自动回滚到安全状态)
        5. ✓ 统一接口 (仿真/真机模式无缝切换)
        6. ✓ 完整日志 (控制台输出 + 文件记录)
        7. ✓ 大模型集成 (支持JSON和自然语言指令解析)
        """)
        
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
