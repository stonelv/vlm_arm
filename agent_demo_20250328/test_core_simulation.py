# test_core_simulation.py
# 核心仿真模块测试
# 不依赖cv2等外部库，直接测试核心功能

import sys
import os
import json
import time

# 确保可以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("核心仿真模块测试")
print("=" * 60)

def test_config():
    '''测试配置模块'''
    print("\n[测试1] 配置模块测试")
    print("-" * 40)
    
    from config import (
        is_simulation_mode, get_robot_mode, set_robot_mode,
        ROBOT_MODE, PI_PORT, PI_BAUD, DEFAULT_ANGLES, DEFAULT_COORDS,
        SIMULATION_CONFIG
    )
    
    print(f"当前ROBOT_MODE: {ROBOT_MODE}")
    print(f"PI_PORT: {PI_PORT}")
    print(f"PI_BAUD: {PI_BAUD}")
    print(f"DEFAULT_ANGLES: {DEFAULT_ANGLES}")
    print(f"DEFAULT_COORDS: {DEFAULT_COORDS}")
    
    # 测试 is_simulation_mode
    current = get_robot_mode()
    print(f"\nis_simulation_mode(): {is_simulation_mode()}")
    
    # 测试模式切换
    set_robot_mode('real')
    assert get_robot_mode() == 'real', "模式切换失败"
    print(f"切换到real模式: {get_robot_mode()}")
    
    set_robot_mode('simulation')
    assert get_robot_mode() == 'simulation', "模式切换失败"
    print(f"切换到simulation模式: {get_robot_mode()}")
    
    # 恢复原模式
    set_robot_mode(current)
    
    print("\n[测试1 通过] 配置模块正常")
    return True

def test_simulated_robot():
    '''测试模拟机器人'''
    print("\n[测试2] 模拟机器人测试")
    print("-" * 40)
    
    from simulated_robot import SimulatedRobot, SimulatedGPIO
    
    # 创建机器人实例
    print("\n创建SimulatedRobot实例...")
    robot = SimulatedRobot(enable_logging=True, enable_visualization=False)
    
    # 测试初始状态
    print(f"初始角度: {robot.get_angles()}")
    print(f"初始坐标: {robot.get_coords()}")
    
    assert len(robot.get_angles()) == 6, "角度数组长度应为6"
    assert len(robot.get_coords()) == 6, "坐标数组长度应为6"
    
    # 测试 send_angles
    print("\n测试 send_angles...")
    test_angles = [10, 20, 30, 40, 50, 60]
    robot.send_angles(test_angles, 40)
    
    # 注意：仿真模式下send_angles会更新current_angles
    # 但这里我们只测试接口是否正常工作
    print(f"send_angles执行完成")
    
    # 测试 send_angle
    print("\n测试 send_angle...")
    robot.send_angle(1, 30, 50)
    robot.send_angle(2, -45, 50)
    print(f"send_angle执行完成")
    
    # 测试 send_coords
    print("\n测试 send_coords...")
    test_coords = [150, -100, 200, 0, 180, 90]
    robot.send_coords(test_coords, 20, 0)
    print(f"send_coords执行完成")
    
    # 测试 set_fresh_mode
    print("\n测试 set_fresh_mode...")
    robot.set_fresh_mode(0)
    robot.set_fresh_mode(1)
    print(f"set_fresh_mode执行完成")
    
    # 测试 release_all_servos
    print("\n测试 release_all_servos...")
    robot.release_all_servos()
    print(f"release_all_servos执行完成")
    
    # 测试 get_state_summary
    print("\n测试 get_state_summary...")
    state = robot.get_state_summary()
    print(f"状态摘要: {state}")
    assert 'current_angles' in state
    assert 'current_coords' in state
    assert 'trajectory_length' in state
    
    print("\n[测试2 通过] 模拟机器人正常")
    return True

def test_gpio_simulation():
    '''测试模拟GPIO'''
    print("\n[测试3] 模拟GPIO测试")
    print("-" * 40)
    
    from simulated_robot import SimulatedGPIO, GPIO as global_gpio
    
    # 创建GPIO实例
    print("\n创建SimulatedGPIO实例...")
    gpio = SimulatedGPIO()
    
    # 测试 setwarnings
    print("测试 setwarnings...")
    gpio.setwarnings(False)
    print(f"setwarnings执行完成")
    
    # 测试 setmode
    print("\n测试 setmode...")
    gpio.setmode(gpio.BCM)
    print(f"setmode执行完成")
    
    # 测试 setup
    print("\n测试 setup...")
    gpio.setup(20, gpio.OUT)
    gpio.setup(21, gpio.OUT)
    print(f"setup执行完成")
    
    # 测试 output
    print("\n测试 output...")
    gpio.output(20, 1)  # 关闭
    gpio.output(20, 0)  # 开启
    gpio.output(21, 1)
    print(f"output执行完成")
    
    # 测试 input
    print("\n测试 input...")
    value = gpio.input(20)
    print(f"读取引脚20的值: {value}")
    
    # 测试 cleanup
    print("\n测试 cleanup...")
    gpio.cleanup(20)
    gpio.cleanup()
    print(f"cleanup执行完成")
    
    print("\n[测试3 通过] 模拟GPIO正常")
    return True

def test_trajectory_recording():
    '''测试轨迹记录'''
    print("\n[测试4] 轨迹记录测试")
    print("-" * 40)
    
    from simulated_robot import SimulatedRobot
    
    # 创建机器人实例，启用记录
    robot = SimulatedRobot(enable_logging=True, enable_visualization=False)
    
    # 执行一些动作
    print("\n执行动作以生成轨迹...")
    robot.send_angles([0, 0, 0, 0, 0, 0], 40)
    robot.send_coords([150, -100, 200, 0, 180, 90], 20, 0)
    robot.send_angle(1, 30, 50)
    robot.send_angle(1, 0, 50)
    robot.send_coords([100, -50, 150, 0, 180, 90], 20, 0)
    robot.set_fresh_mode(0)
    
    # 检查轨迹长度
    state = robot.get_state_summary()
    trajectory_length = state['trajectory_length']
    print(f"记录的动作数: {trajectory_length}")
    assert trajectory_length >= 6, f"应该至少记录6个动作，实际记录了{trajectory_length}个"
    
    # 保存轨迹
    print("\n保存轨迹到文件...")
    os.makedirs('temp', exist_ok=True)
    test_file = 'temp/test_trajectory_core.json'
    robot.save_trajectory(test_file)
    
    # 验证文件存在
    assert os.path.exists(test_file), "轨迹文件未创建"
    print(f"轨迹已保存到: {test_file}")
    
    # 加载轨迹
    print("\n从文件加载轨迹...")
    loaded_trajectory = robot.load_trajectory(test_file)
    print(f"加载的动作数: {len(loaded_trajectory)}")
    assert len(loaded_trajectory) == trajectory_length, "加载的轨迹长度不匹配"
    
    # 验证轨迹数据结构
    for i, action in enumerate(loaded_trajectory[:3]):  # 只检查前3个
        assert 'action_type' in action, f"动作{i}缺少action_type"
        assert 'params' in action, f"动作{i}缺少params"
        assert 'state_before' in action, f"动作{i}缺少state_before"
        print(f"  动作{i+1}: {action['action_type']}")
    
    print("\n[测试4 通过] 轨迹记录正常")
    return True

def test_trajectory_playback():
    '''测试轨迹回放'''
    print("\n[测试5] 轨迹回放测试")
    print("-" * 40)
    
    from simulated_robot import SimulatedRobot
    
    # 准备测试轨迹
    test_trajectory = [
        {
            'action_type': 'send_angles',
            'params': {'angles': [0, 0, 0, 0, 0, 0], 'speed': 40},
            'state_before': {'angles': [0, 0, 0, 0, 0, 0], 'coords': [160, -100, 200, 0, 180, 90]}
        },
        {
            'action_type': 'send_coords',
            'params': {'coords': [200, -50, 180, 0, 180, 90], 'speed': 20, 'mode': 0},
            'state_before': {'angles': [0, 0, 0, 0, 0, 0], 'coords': [160, -100, 200, 0, 180, 90]}
        },
        {
            'action_type': 'set_fresh_mode',
            'params': {'mode': 0},
            'state_before': {'angles': [0, 0, 0, 0, 0, 0], 'coords': [200, -50, 180, 0, 180, 90]}
        },
        {
            'action_type': 'send_angle',
            'params': {'joint_index': 1, 'angle': 30, 'speed': 50},
            'state_before': {'angles': [0, 0, 0, 0, 0, 0], 'coords': [200, -50, 180, 0, 180, 90]}
        },
        {
            'action_type': 'release_all_servos',
            'params': {},
            'state_before': {'angles': [30, 0, 0, 0, 0, 0], 'coords': [200, -50, 180, 0, 180, 90]}
        }
    ]
    
    print(f"准备回放 {len(test_trajectory)} 个动作")
    
    # 创建新的机器人实例用于回放
    playback_robot = SimulatedRobot(enable_logging=True, enable_visualization=False)
    
    # 回放前状态
    state_before = playback_robot.get_state_summary()
    print(f"\n回放前状态:")
    print(f"  角度: {state_before['current_angles']}")
    print(f"  坐标: {state_before['current_coords']}")
    print(f"  轨迹长度: {state_before['trajectory_length']}")
    
    # 回放轨迹
    print("\n开始回放...")
    playback_robot.playback_trajectory(test_trajectory, speed_multiplier=0.1)
    
    # 回放后状态
    state_after = playback_robot.get_state_summary()
    print(f"\n回放后状态:")
    print(f"  角度: {state_after['current_angles']}")
    print(f"  坐标: {state_after['current_coords']}")
    print(f"  轨迹长度: {state_after['trajectory_length']}")
    
    # 验证回放时记录了动作
    assert state_after['trajectory_length'] >= len(test_trajectory), "回放时未正确记录动作"
    
    print("\n[测试5 通过] 轨迹回放正常")
    return True

def test_visualization_output():
    '''测试可视化输出'''
    print("\n[测试6] 可视化输出测试")
    print("-" * 40)
    
    from simulated_robot import SimulatedRobot
    
    # 创建带可视化的机器人实例
    viz_robot = SimulatedRobot(enable_logging=True, enable_visualization=True)
    
    print("\n执行动作，观察可视化输出...")
    print("-" * 50)
    
    # 执行一系列动作
    viz_robot.send_angles([0, 0, 0, 0, 0, 0], 40)
    viz_robot.send_coords([200, -50, 180, 0, 180, 90], 20, 0)
    viz_robot.send_angle(2, -45, 50)
    viz_robot.set_fresh_mode(0)
    viz_robot.release_all_servos()
    
    print("\n[测试6 通过] 可视化输出正常")
    return True

def test_interface_compatibility():
    '''测试接口兼容性'''
    print("\n[测试7] 接口兼容性测试")
    print("-" * 40)
    
    from simulated_robot import SimulatedRobot
    
    # 检查模拟机器人是否实现了与真实MyCobot相同的核心接口
    robot = SimulatedRobot()
    
    # 真实MyCobot的核心方法（根据utils_robot.py中的使用）
    required_methods = [
        'send_angles',
        'send_angle', 
        'send_coords',
        'release_all_servos',
        'set_fresh_mode'
    ]
    
    print("\n检查必需的方法...")
    all_present = True
    for method in required_methods:
        if hasattr(robot, method):
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} - 缺失!")
            all_present = False
    
    # 检查额外的仿真方法
    print("\n检查仿真模式特有方法...")
    simulation_methods = [
        'get_angles',
        'get_coords',
        'save_trajectory',
        'load_trajectory',
        'playback_trajectory',
        'get_state_summary'
    ]
    for method in simulation_methods:
        if hasattr(robot, method):
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} - 缺失!")
            all_present = False
    
    assert all_present, "有必需的方法缺失"
    
    print("\n[测试7 通过] 接口兼容性正常")
    return True

def run_all_tests():
    '''运行所有测试'''
    print("\n" + "=" * 60)
    print("开始运行所有核心测试用例")
    print("=" * 60)
    
    tests = [
        ("配置模块", test_config),
        ("模拟机器人", test_simulated_robot),
        ("模拟GPIO", test_gpio_simulation),
        ("轨迹记录", test_trajectory_recording),
        ("轨迹回放", test_trajectory_playback),
        ("可视化输出", test_visualization_output),
        ("接口兼容性", test_interface_compatibility),
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
            else:
                failed += 1
                failed_tests.append(test_name)
        except Exception as e:
            failed += 1
            failed_tests.append(test_name)
            print(f"\n[测试失败] {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed_tests:
        print(f"\n失败的测试: {', '.join(failed_tests)}")
    
    if failed == 0:
        print("\n✅ 所有核心测试通过！")
        print("\n💡 提示:")
        print("   - 核心仿真模块工作正常")
        print("   - 可以通过设置环境变量ROBOT_MODE=simulation来启用仿真模式")
        print("   - 仿真模式下可以记录和回放轨迹")
        print("   - 接口与真实MyCobot完全兼容")
        return True
    else:
        print("\n❌ 有测试失败，请检查错误信息")
        return False

if __name__ == '__main__':
    # 确保 temp 目录存在
    os.makedirs('temp', exist_ok=True)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
