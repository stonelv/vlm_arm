"""
拖拽示教-可复用技能库 使用示例
演示如何录制轨迹、创建技能、保存加载、组合执行
"""

print("="*70)
print("拖拽示教-可复用技能库 使用示例")
print("="*70)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_library import (
    SkillLibrary, SkillExecutor, SkillComposer,
    EnhancedRecorder, TrajectoryProcessor,
    Skill, CompositeSkill, TrajectoryPoint, SkillParameter,
    EventType, create_demo_skills, HAS_PYMYCOBOT, HAS_GPIO
)


def demo_1_create_and_save_skills():
    """
    示例1: 创建并保存基础技能
    """
    print("\n" + "-"*70)
    print("示例1: 创建并保存基础技能")
    print("-"*70)
    
    library = SkillLibrary()
    
    print("\n创建演示技能...")
    pick_skill, move_skill, place_skill, composite = create_demo_skills(library)
    
    print(f"\n已创建技能:")
    print(f"  - pick_up: {pick_skill.description}")
    print(f"  - move: {move_skill.description}")
    print(f"  - place_down: {place_skill.description}")
    print(f"  - pick_and_place (复合): {composite.description}")
    
    print("\n查看技能参数:")
    for param in pick_skill.parameters:
        print(f"  - {param.name}: {param.description} (默认: {param.default_value})")
    
    return library


def demo_2_load_and_search_skills(library: SkillLibrary):
    """
    示例2: 加载和搜索技能
    """
    print("\n" + "-"*70)
    print("示例2: 加载和搜索技能")
    print("-"*70)
    
    all_skills = library.list_all_skills()
    print(f"\n技能库中的技能:")
    print(f"  基础技能: {all_skills['primitive']}")
    print(f"  复合技能: {all_skills['composite']}")
    
    print("\n搜索关键词 'pick':")
    results = library.search_skills("pick")
    for skill in results:
        print(f"  - {skill.name}: {skill.description}")
    
    skill = library.load_skill("pick_up")
    if skill:
        print(f"\n加载技能 'pick_up' 成功!")
        print(f"  轨迹点数: {len(skill.trajectory)}")
        print(f"  标准化时长: {skill.normalized_duration}s")


def demo_3_trajectory_processing():
    """
    示例3: 轨迹平滑与时间归一化
    """
    print("\n" + "-"*70)
    print("示例3: 轨迹平滑与时间归一化")
    print("-"*70)
    
    print("\n创建示例轨迹...")
    raw_trajectory = [
        TrajectoryPoint(timestamp=0.0, angles=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.1, angles=[1.2, -5.3, 4.1, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.2, angles=[2.8, -12.1, 9.5, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.3, angles=[3.1, -18.9, 15.2, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.4, angles=[5.2, -25.1, 22.3, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.5, angles=[4.8, -30.2, 28.9, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.6, angles=[5.0, -35.5, 35.1, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.7, angles=[5.1, -40.0, 40.0, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.8, angles=[5.0, -45.2, 45.3, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=0.9, angles=[5.1, -49.8, 49.7, 0.0, 0.0, 0.0]),
        TrajectoryPoint(timestamp=1.0, angles=[5.0, -50.0, 50.0, 0.0, 0.0, 0.0]),
    ]
    
    print(f"原始轨迹点数: {len(raw_trajectory)}")
    print(f"原始时长: {raw_trajectory[-1].timestamp - raw_trajectory[0].timestamp}s")
    
    print("\n应用移动平均平滑 (窗口大小=3)...")
    smoothed = TrajectoryProcessor.moving_average_smooth(raw_trajectory, window_size=3)
    print(f"平滑后轨迹点数: {len(smoothed)}")
    
    print("\n应用时间归一化 (目标时长=5.0s, 点数=50)...")
    normalized = TrajectoryProcessor.time_normalize(
        smoothed, target_duration=5.0, num_points=50
    )
    print(f"归一化后轨迹点数: {len(normalized)}")
    print(f"归一化后时长: {normalized[-1].timestamp - normalized[0].timestamp}s")
    
    print("\n提取关键帧 (角度阈值=5.0度)...")
    keyframes = TrajectoryProcessor.extract_keyframes(normalized, angle_threshold=5.0)
    print(f"关键帧数量: {len(keyframes)}")


def demo_4_composite_skill(library: SkillLibrary):
    """
    示例4: 创建和执行复合技能
    """
    print("\n" + "-"*70)
    print("示例4: 创建'拿起-移动-放下'复合技能")
    print("-"*70)
    
    composer = SkillComposer(library)
    
    print("\n创建自定义复合技能...")
    custom_composite = composer.create_composite_skill(
        name="pick_and_place_custom",
        description="自定义的拾取-移动-放置复合技能",
        sub_skills=[
            ("pick_up", {"speed": 0.8, "height_offset": 0.0}),
            ("move", {"speed": 1.0}),
            ("place_down", {"speed": 0.8, "height_offset": 0.0})
        ]
    )
    
    library.save_skill(custom_composite)
    
    print(f"\n复合技能 'pick_and_place_custom' 包含的子技能:")
    for sub_name, params in custom_composite.sub_skills:
        print(f"  - {sub_name}: 参数 = {params}")
    
    return custom_composite


def demo_5_execute_skills(library: SkillLibrary):
    """
    示例5: 执行技能 (模拟模式)
    """
    print("\n" + "-"*70)
    print("示例5: 执行技能 (模拟模式)")
    print("-"*70)
    
    executor = SkillExecutor(library=library)
    
    print("\n" + "*"*50)
    print("注意: 当前环境没有连接真实机械臂")
    print("以下是模拟执行演示")
    print("*"*50)
    
    print("\n执行基础技能 'pick_up'...")
    pick_skill = library.load_skill("pick_up")
    if pick_skill:
        executor.execute_skill(
            pick_skill,
            params={"speed": 1.0},
            use_smoothed=True
        )
    
    print("\n执行复合技能 'pick_and_place'...")
    composite_skill = library.load_skill("pick_and_place")
    if composite_skill and isinstance(composite_skill, CompositeSkill):
        executor.execute_composite_skill(
            composite_skill,
            global_params={"global_speed": 1.0}
        )


def demo_6_recorder_workflow():
    """
    示例6: 录制器工作流程 (概念演示)
    """
    print("\n" + "-"*70)
    print("示例6: 录制器工作流程 (概念演示)")
    print("-"*70)
    
    print("\n录制器使用流程:")
    print("""
    1. 初始化录制器:
       recorder = EnhancedRecorder(mc=机械臂实例)
    
    2. 开始录制:
       recorder.start_recording(sample_interval=0.1)
    
    3. 录制过程中记录吸泵事件:
       recorder.record_pump_event(EventType.PUMP_ON)   # 开启吸泵
       recorder.record_pump_event(EventType.PUMP_OFF)  # 关闭吸泵
    
    4. 停止录制:
       trajectory = recorder.stop_recording()
    
    5. 从录制数据创建技能:
       skill = recorder.create_skill_from_recording(
           name="my_custom_skill",
           description="这是我录制的自定义技能",
           smooth=True,
           normalize=True,
           smooth_method="moving_average",
           target_duration=5.0
       )
    
    6. 保存到技能库:
       library.save_skill(skill)
    """)
    
    print("\n提示: 录制的轨迹会自动:")
    print("  - 记录每个点的关节角度和坐标")
    print("  - 记录吸泵开关事件")
    print("  - 应用轨迹平滑处理")
    print("  - 进行时间归一化")
    print("  - 自动提取起始/目标角度作为参数")


def main():
    """
    主函数: 运行所有示例
    """
    print("\n检查环境...")
    print(f"  pymycobot 可用: {HAS_PYMYCOBOT}")
    print(f"  RPi.GPIO 可用: {HAS_GPIO}")
    
    if not HAS_PYMYCOBOT or not HAS_GPIO:
        print("\n警告: 在没有真实硬件的环境中运行")
        print("      所有机械臂操作将以模拟模式执行")
    
    library = demo_1_create_and_save_skills()
    
    demo_2_load_and_search_skills(library)
    
    demo_3_trajectory_processing()
    
    demo_4_composite_skill(library)
    
    demo_5_execute_skills(library)
    
    demo_6_recorder_workflow()
    
    print("\n" + "="*70)
    print("所有示例运行完成!")
    print("="*70)
    print("""
总结:
1. 技能库可以创建、保存、加载、搜索技能
2. 支持轨迹平滑 (移动平均、Savitzky-Golay) 和时间归一化
3. 技能可以参数化 (起点、目标、速度、高度等)
4. 复合技能可以组合多个子技能按顺序执行
5. 'pick_and_place' 复合技能演示了:
   - pick_up: 下降 → 开吸泵 → 上升
   - move: 水平移动
   - place_down: 下降 → 关吸泵 → 上升

运行方式:
- 交互模式: python skill_library.py
- 示例演示: python example_skill_usage.py
""")


if __name__ == "__main__":
    main()
