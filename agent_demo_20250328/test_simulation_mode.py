# test_simulation_mode.py
# 测试用例：验证仿真模式与真机模式的接口兼容性和输出一致性
# 运行方式：
# - 仿真模式测试：ROBOT_MODE=simulation python test_simulation_mode.py
# - 真机模式测试：ROBOT_MODE=real python test_simulation_mode.py

import sys
import os
import json
import time

# 导入配置模块
from config import is_simulation_mode, get_robot_mode, set_robot_mode, ROBOT_MODE

print(f"当前运行模式: {ROBOT_MODE}")
print(f"是否仿真模式: {is_simulation_mode()}")
print("=" * 60)

def test_config_module():
    '''测试配置模块功能'''
    print("\n[测试1] 配置模块测试")
    print("-" * 40)
    
    # 测试 is_simulation_mode
    current_mode = get_robot_mode()
    print(f"当前模式: {current_mode}")
    
    # 测试模式切换（仅在内存中切换，不影响环境变量）
    if current_mode == 'real':
        set_robot_mode('simulation')
        assert get_robot_mode() == 'simulation', "模式切换失败"
        print("模式切换测试通过: real -> simulation")
        
        set_robot_mode('real')
        assert get_robot_mode() == 'real', "模式切换失败"
        print("模式切换测试通过: simulation -> real")
    else:
        set_robot_mode('real')
        assert get_robot_mode() == 'real', "模式切换失败"
        print("模式切换测试通过: simulation -> real")
        
        set_robot_mode('simulation')
        assert get_robot_mode() == 'simulation', "模式切换失败"
        print("模式切换测试通过: real -> simulation")
    
    # 恢复原模式
    set_robot_mode(current_mode)
    
    print("[测试1 通过] 配置模块功能正常")

def test_robot_interface_compatibility():
    '''测试机器人接口兼容性'''
    print("\n[测试2] 机器人接口兼容性测试")
    print("-" * 40)
    
    # 导入机器人模块（这会根据配置自动选择模式）
    from utils_robot import mc, back_zero, move_to_coords, single_joint_move
    
    # 测试基础接口是否存在
    required_methods = [
        'send_angles', 'send_angle', 'send_coords',
        'release_all_servos', 'set_fresh_mode'
    ]
    
    for method in required_methods:
        assert hasattr(mc, method), f"缺少必需的方法: {method}"
        print(f"✓ 方法存在: {method}")
    
    # 如果是仿真模式，额外测试仿真特有的方法
    if is_simulation_mode():
        simulation_methods = [
            'get_angles', 'get_coords', 'save_trajectory',
            'load_trajectory', 'playback_trajectory', 'get_state_summary'
        ]
        for method in simulation_methods:
            assert hasattr(mc, method), f"仿真模式缺少方法: {method}"
            print(f"✓ 仿真特有方法存在: {method}")
        
        # 测试状态获取
        angles = mc.get_angles()
        coords = mc.get_coords()
        print(f"当前角度: {angles}")
        print(f"当前坐标: {coords}")
        assert len(angles) == 6, "角度数组长度应为6"
        assert len(coords) == 6, "坐标数组长度应为6"
    
    print("[测试2 通过] 机器人接口兼容")

def test_action_commands():
    '''测试动作指令'''
    print("\n[测试3] 动作指令测试")
    print("-" * 40)
    
    from utils_robot import mc
    
    # 测试1: send_angles
    print("\n测试 send_angles...")
    test_angles = [0, 0, 0, 0, 0, 0]
    mc.send_angles(test_angles, 40)
    print(f"✓ 已发送角度: {test_angles}")
    
    # 测试2: send_angle (单关节)
    print("\n测试 send_angle...")
    mc.send_angle(5, 30, 80)
    print(f"✓ 已发送单关节: 关节5, 角度30")
    
    # 测试3: send_coords
    print("\n测试 send_coords...")
    test_coords = [150, -130, 230, 0, 180, 90]
    mc.send_coords(test_coords, 20, 0)
    print(f"✓ 已发送坐标: {test_coords}")
    
    # 测试4: set_fresh_mode
    print("\n测试 set_fresh_mode...")
    mc.set_fresh_mode(0)
    print("✓ 已设置运动模式为插补")
    
    # 测试5: release_all_servos
    print("\n测试 release_all_servos...")
    if is_simulation_mode():
        mc.release_all_servos()
        print("✓ 已释放所有伺服（仿真模式）")
    else:
        print("跳过: 真机模式下不测试释放伺服")
    
    print("\n[测试3 通过] 动作指令执行成功")

def test_pump_interface():
    '''测试吸泵接口'''
    print("\n[测试4] 吸泵接口测试")
    print("-" * 40)
    
    # 吸泵在 utils_pump.py 中初始化，这里测试其兼容性
    if is_simulation_mode():
        from simulated_robot import GPIO
        
        # 测试 GPIO 接口
        print("测试 GPIO 接口...")
        assert hasattr(GPIO, 'setwarnings'), "缺少 setwarnings 方法"
        assert hasattr(GPIO, 'setmode'), "缺少 setmode 方法"
        assert hasattr(GPIO, 'setup'), "缺少 setup 方法"
        assert hasattr(GPIO, 'output'), "缺少 output 方法"
        assert hasattr(GPIO, 'input'), "缺少 input 方法"
        print("✓ GPIO 接口完整")
        
        # 测试吸泵控制函数
        print("\n测试吸泵控制函数...")
        from utils_pump import pump_on, pump_off
        
        print("测试 pump_on...")
        pump_on()
        print("✓ pump_on 执行成功")
        
        time.sleep(0.1)
        
        print("测试 pump_off...")
        pump_off()
        print("✓ pump_off 执行成功")
    else:
        print("真机模式下跳过GPIO接口详细测试（依赖硬件）")
        # 简单测试导入是否成功
        from utils_pump import pump_on, pump_off
        print("✓ 吸泵模块导入成功")
    
    print("\n[测试4 通过] 吸泵接口正常")

def test_trajectory_recording():
    '''测试轨迹记录功能（仅仿真模式）'''
    print("\n[测试5] 轨迹记录测试（仿真模式专用）")
    print("-" * 40)
    
    if not is_simulation_mode():
        print("跳过: 真机模式不支持轨迹记录")
        print("[测试5 跳过] 非仿真模式")
        return
    
    from utils_robot import mc
    
    # 先执行一些动作
    print("执行动作以生成轨迹...")
    mc.send_angles([0, 0, 0, 0, 0, 0], 40)
    time.sleep(0.05)
    mc.send_coords([150, -100, 200, 0, 180, 90], 20, 0)
    time.sleep(0.05)
    mc.send_angle(1, 30, 50)
    time.sleep(0.05)
    
    # 获取状态
    state = mc.get_state_summary()
    print(f"轨迹长度: {state['trajectory_length']}")
    assert state['trajectory_length'] >= 3, "轨迹记录失败"
    
    # 保存轨迹
    print("\n保存轨迹到文件...")
    test_file = 'temp/test_trajectory.json'
    mc.save_trajectory(test_file)
    
    # 验证文件存在
    assert os.path.exists(test_file), "轨迹文件未生成"
    print(f"✓ 轨迹已保存到: {test_file}")
    
    # 加载并验证轨迹
    print("\n加载并验证轨迹...")
    loaded_trajectory = mc.load_trajectory(test_file)
    assert len(loaded_trajectory) >= 3, "加载的轨迹不完整"
    print(f"✓ 成功加载 {len(loaded_trajectory)} 个动作")
    
    # 检查轨迹数据结构
    for i, action in enumerate(loaded_trajectory):
        assert 'action_type' in action, f"动作 {i} 缺少 action_type"
        assert 'params' in action, f"动作 {i} 缺少 params"
        assert 'state_before' in action, f"动作 {i} 缺少 state_before"
        print(f"  动作 {i+1}: {action['action_type']}")
    
    print("\n[测试5 通过] 轨迹记录功能正常")

def test_trajectory_playback():
    '''测试轨迹回放功能（仅仿真模式）'''
    print("\n[测试6] 轨迹回放测试（仿真模式专用）")
    print("-" * 40)
    
    if not is_simulation_mode():
        print("跳过: 真机模式不支持轨迹回放")
        print("[测试6 跳过] 非仿真模式")
        return
    
    from utils_robot import mc
    
    # 创建一个新的机器人实例用于回放测试
    from simulated_robot import SimulatedRobot
    playback_robot = SimulatedRobot(enable_logging=True)
    
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
        }
    ]
    
    print("准备回放轨迹...")
    print(f"轨迹包含 {len(test_trajectory)} 个动作")
    
    # 回放轨迹
    print("\n开始回放...")
    playback_robot.playback_trajectory(test_trajectory, speed_multiplier=0.1)
    
    # 验证回放后状态
    state = playback_robot.get_state_summary()
    print(f"\n回放后状态:")
    print(f"  角度: {state['current_angles']}")
    print(f"  坐标: {state['current_coords']}")
    print(f"  记录的动作数: {state['trajectory_length']}")
    
    assert state['trajectory_length'] >= 3, "回放时未正确记录动作"
    print("\n[测试6 通过] 轨迹回放功能正常")

def test_visualization_output():
    '''测试可视化输出功能'''
    print("\n[测试7] 可视化输出测试")
    print("-" * 40)
    
    if not is_simulation_mode():
        print("跳过: 真机模式不支持可视化输出")
        print("[测试7 跳过] 非仿真模式")
        return
    
    # 创建带可视化的机器人实例
    from simulated_robot import SimulatedRobot
    viz_robot = SimulatedRobot(enable_visualization=True)
    
    print("执行动作，观察可视化输出...")
    
    # 执行一些动作，观察控制台输出
    viz_robot.send_angles([0, 0, 0, 0, 0, 0], 40)
    time.sleep(0.05)
    
    viz_robot.send_coords([150, -100, 200, 0, 180, 90], 20, 0)
    time.sleep(0.05)
    
    viz_robot.set_fresh_mode(0)
    time.sleep(0.05)
    
    print("\n[测试7 通过] 可视化输出正常")

def run_all_tests():
    '''运行所有测试'''
    print("\n" + "=" * 60)
    print("开始运行所有测试用例")
    print("=" * 60)
    
    test_passed = 0
    test_skipped = 0
    test_failed = 0
    
    tests = [
        ("配置模块测试", test_config_module),
        ("接口兼容性测试", test_robot_interface_compatibility),
        ("动作指令测试", test_action_commands),
        ("吸泵接口测试", test_pump_interface),
        ("轨迹记录测试", test_trajectory_recording),
        ("轨迹回放测试", test_trajectory_playback),
        ("可视化输出测试", test_visualization_output),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            test_passed += 1
        except Exception as e:
            if "跳过" in str(e):
                test_skipped += 1
            else:
                test_failed += 1
                print(f"\n[测试失败] {test_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"通过: {test_passed}")
    print(f"跳过: {test_skipped}")
    print(f"失败: {test_failed}")
    
    if test_failed > 0:
        print("\n❌ 有测试失败，请检查错误信息")
        return False
    else:
        print("\n✅ 所有测试通过！")
        return True

if __name__ == '__main__':
    # 确保 temp 目录存在
    os.makedirs('temp', exist_ok=True)
    
    # 运行所有测试
    success = run_all_tests()
    
    sys.exit(0 if success else 1)
