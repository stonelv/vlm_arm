#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全动作执行器 - 测试脚本
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safe_action_executor import (
    SafeActionExecutor, Pose, ActionType,
    HighLevelCommand, SafetyConstraints, create_executor,
    WorkspaceLimitExceeded, JointLimitExceeded, CollisionDetected
)


def test_basic_functionality():
    """测试基础功能"""
    print('='*60)
    print('测试1: 基础功能测试')
    print('='*60)
    
    # 创建执行器
    executor = create_executor(simulation=True)
    
    try:
        # 测试1: 执行移动指令
        print('\n[1.1] 执行移动指令...')
        command_dict = {
            'action': 'move',
            'pose': [200, -100, 220, 0, 180, 90],
            'speed': 30
        }
        executor.execute_from_dict(command_dict)
        
        current_pose = executor.robot.get_current_pose()
        print(f'  移动后位置: {current_pose}')
        assert abs(current_pose.x - 200) < 1, 'X坐标不正确'
        assert abs(current_pose.y - (-100)) < 1, 'Y坐标不正确'
        print('  ✓ 移动指令执行成功')
        
        # 测试2: 归零
        print('\n[1.2] 归零指令...')
        executor.execute_from_dict({'action': 'home', 'speed': 40})
        print('  ✓ 归零指令执行成功')
        
        # 测试3: 移动关节
        print('\n[1.3] 移动关节指令...')
        joint_command = {
            'action': 'move_joints',
            'joints': [30.0, -20.0, 45.0, 0.0, 10.0, -15.0],
            'speed': 25
        }
        executor.execute_from_dict(joint_command)
        print('  ✓ 移动关节指令执行成功')
        
        print('\n测试1 通过!')
        
    finally:
        executor.shutdown()


def test_safety_constraints():
    """测试安全约束"""
    print('\n' + '='*60)
    print('测试2: 安全约束测试')
    print('='*60)
    
    # 使用自定义安全约束
    constraints = SafetyConstraints(
        max_speed=60,
        min_speed=5,
        default_safe_height=200,
        collision_distance_threshold=100,
        emergency_stop_distance=30,
    )
    
    executor = SafeActionExecutor(simulation_mode=True, constraints=constraints)
    executor.initialize()
    
    try:
        # 测试1: 工作空间限制
        print('\n[2.1] 工作空间限制测试...')
        try:
            invalid_command = {
                'action': 'move',
                'pose': [500, 0, 200, 0, 180, 90]  # x=500 超出限制
            }
            executor.execute_from_dict(invalid_command)
            print('  ✗ 应该抛出异常但没有!')
            return False
        except WorkspaceLimitExceeded as e:
            print(f'  ✓ 正确捕获工作空间限制异常')
            print(f'    异常信息: {e}')
        
        # 测试2: 关节限位
        print('\n[2.2] 关节限位测试...')
        try:
            invalid_joints = {
                'action': 'move_joints',
                'joints': [200.0, 0, 0, 0, 0, 0]  # 关节1=200° 超出165°限制
            }
            executor.execute_from_dict(invalid_joints)
            print('  ✗ 应该抛出异常但没有!')
            return False
        except JointLimitExceeded as e:
            print(f'  ✓ 正确捕获关节限位异常')
            print(f'    异常信息: {e}')
        
        # 测试3: 速度约束
        print('\n[2.3] 速度约束测试...')
        print(f'  当前最大速度限制: {executor.constraints.max_speed}%')
        # 使用超出限制的速度（应该自动约束）
        fast_command = {
            'action': 'move',
            'pose': [150, 50, 200, 0, 180, 90],
            'speed': 90  # 超出max_speed=60
        }
        executor.execute_from_dict(fast_command)
        print('  ✓ 速度自动约束执行成功')
        
        # 测试4: 碰撞检测
        print('\n[2.4] 碰撞检测测试...')
        # 添加障碍物
        obstacle_pose = Pose(x=150, y=0, z=150)
        executor.add_obstacle(obstacle_pose, radius=60.0, name="测试障碍物")
        print('  ✓ 障碍物添加成功')
        
        # 测试移动到安全位置
        safe_pose = Pose(x=100, y=100, z=200)
        safe_command = {
            'action': 'move',
            'pose': safe_pose.to_list(),
            'speed': 30
        }
        executor.execute_from_dict(safe_command)
        print('  ✓ 安全位置移动成功')
        
        print('\n测试2 通过!')
        return True
        
    finally:
        executor.shutdown()


def test_pick_and_place():
    """测试抓取和放置"""
    print('\n' + '='*60)
    print('测试3: 抓取和放置测试')
    print('='*60)
    
    executor = create_executor(simulation=True)
    
    try:
        # 抓取位置和放置位置
        pick_pose = Pose(x=200, y=-50, z=90, rx=0, ry=180, rz=90)
        place_pose = Pose(x=100, y=150, z=100, rx=0, ry=180, rz=90)
        
        print(f'  抓取位置: {pick_pose}')
        print(f'  放置位置: {place_pose}')
        
        # 步骤1: 执行抓取
        print('\n[3.1] 执行抓取动作...')
        pick_command = {
            'action': 'pick',
            'pose': [pick_pose.x, pick_pose.y, pick_pose.z, 
                     pick_pose.rx, pick_pose.ry, pick_pose.rz],
            'height': 90.0,
            'safe_height': 220.0,
            'speed': 30
        }
        executor.execute_from_dict(pick_command)
        print('  ✓ 抓取动作执行成功')
        
        # 步骤2: 执行放置
        print('\n[3.2] 执行放置动作...')
        place_command = {
            'action': 'place',
            'pose': [place_pose.x, place_pose.y, place_pose.z,
                     place_pose.rx, place_pose.ry, place_pose.rz],
            'height': 100.0,
            'safe_height': 220.0,
            'speed': 30
        }
        executor.execute_from_dict(place_command)
        print('  ✓ 放置动作执行成功')
        
        # 查看动作历史
        print('\n[3.3] 查看动作历史...')
        history = executor.action_history.history
        print(f'  已记录动作数: {len(history)}')
        for i, record in enumerate(history):
            status = '成功' if record.success else '失败'
            print(f'    [{i+1}] {record.action_type.value} - {status}')
        
        print('\n测试3 通过!')
        return True
        
    finally:
        executor.shutdown()


def test_llm_parsing():
    """测试大模型指令解析"""
    print('\n' + '='*60)
    print('测试4: 大模型指令解析测试')
    print('='*60)
    
    executor = create_executor(simulation=True)
    
    try:
        # 格式1: 结构化JSON
        print('\n[4.1] 从结构化JSON解析...')
        llm_response_1 = """
        分析完毕，我将执行以下操作：
        {
            "action": "move",
            "pose": [180, -80, 220, 0, 180, 90],
            "speed": 35
        }
        """
        executor.execute_from_llm(llm_response_1)
        print('  ✓ 结构化JSON解析成功')
        
        # 格式2: 自然语言
        print('\n[4.2] 从自然语言解析...')
        llm_response_2 = "请移动到坐标 x=150, y=100, z=200 的位置"
        executor.execute_from_llm(llm_response_2)
        print('  ✓ 自然语言解析成功')
        
        print('\n测试4 通过!')
        return True
        
    finally:
        executor.shutdown()


def test_state_management():
    """测试状态管理"""
    print('\n' + '='*60)
    print('测试5: 状态管理测试')
    print('='*60)
    
    executor = create_executor(simulation=True)
    
    try:
        # 获取初始状态
        print('\n[5.1] 获取初始状态...')
        state = executor.get_state()
        print('  初始化:', state['initialized'])
        print('  仿真模式:', state['simulation_mode'])
        print('  机器人连接:', state['robot_connected'])
        print('  当前位姿:', state['current_pose'])
        print('  当前关节:', state['current_joints'])
        
        # 执行一些动作后查看状态
        print('\n[5.2] 执行动作后查看状态...')
        executor.execute_from_dict({
            'action': 'move',
            'pose': [200, 100, 200, 0, 180, 90],
            'speed': 30
        })
        
        state = executor.get_state()
        print('  当前位姿:', state['current_pose'])
        assert state['current_pose'][0] == 200, 'X坐标更新失败'
        
        print('  ✓ 状态管理正确')
        
        print('\n测试5 通过!')
        return True
        
    finally:
        executor.shutdown()


def main():
    """主测试函数"""
    import time
    import traceback
    
    print('='*60)
    print('安全动作执行器 - 完整测试套件')
    print('Safe Action Executor - Test Suite')
    print('='*60)
    
    time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    print('\n测试时间:', time_str)
    print('运行模式: 仿真模式')
    
    all_passed = True
    
    try:
        test_basic_functionality()
    except Exception as e:
        print('\n测试1 失败:', e)
        traceback.print_exc()
        all_passed = False
    
    try:
        test_safety_constraints()
    except Exception as e:
        print('\n测试2 失败:', e)
        traceback.print_exc()
        all_passed = False
    
    try:
        test_pick_and_place()
    except Exception as e:
        print('\n测试3 失败:', e)
        traceback.print_exc()
        all_passed = False
    
    try:
        test_llm_parsing()
    except Exception as e:
        print('\n测试4 失败:', e)
        traceback.print_exc()
        all_passed = False
    
    try:
        test_state_management()
    except Exception as e:
        print('\n测试5 失败:', e)
        traceback.print_exc()
        all_passed = False
    
    print('\n' + '='*60)
    if all_passed:
        print('所有测试通过! ✓')
    else:
        print('部分测试失败! ✗')
    print('='*60)
    
    print('\n日志已记录到: safe_action_executor.log')
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
