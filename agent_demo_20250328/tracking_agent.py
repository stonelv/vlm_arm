# tracking_agent.py
# 实时目标追踪智能体入口
# 功能：驱动机械臂根据摄像头实时画面追踪移动的人脸或指定物品

print('\n实时目标追踪智能体')
print('机械臂 + 视觉追踪 + 语音交互 = 具身智能追踪系统\n')

import sys
import os

from utils_tracking import RealTimeTracker

def main():
    print('=== 实时目标追踪系统 ===')
    print('功能说明:')
    print('1. 自动追踪视野内优先级最高的目标 (人脸 > 红盒子 > 绿盒子 > 蓝盒子 > 黄盒子)')
    print('2. 保持机械臂末端与目标的恒定相对距离')
    print('3. 支持目标短暂遮挡时的运动预测')
    print('4. 支持多目标优先级自动切换')
    print('5. 支持语音指令动态切换追踪对象\n')
    
    print('语音指令示例:')
    print('  - "跟着红色盒子" / "追踪那个人脸"')
    print('  - "停止追踪" / "开始追踪"')
    print('  - "恢复自动模式" (回到优先级自动选择)\n')
    
    print('键盘控制:')
    print('  - 按 q 键退出系统')
    print('  - 按 空格键 开始/暂停追踪\n')
    
    try:
        tracker = RealTimeTracker()
        tracker.initialize()
        tracker.start()
        
    except KeyboardInterrupt:
        print('\n用户中断')
    except Exception as e:
        print(f'系统错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        print('系统已关闭')

if __name__ == '__main__':
    main()
