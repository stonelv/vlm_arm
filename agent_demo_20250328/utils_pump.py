# utils_pump.py
# 同济子豪兄 2024-5-22
# GPIO引脚、吸泵相关函数
# 支持真机/仿真模式切换

print('导入吸泵控制模块')

from config import is_simulation_mode
import time

# 根据模式选择GPIO接口
if is_simulation_mode():
    print('[模式检测] 仿真模式，使用模拟GPIO接口')
    from simulated_robot import GPIO
else:
    print('[模式检测] 真机模式，使用真实RPi.GPIO接口')
    import RPi.GPIO as GPIO

# 初始化GPIO
GPIO.setwarnings(False)   # 不打印 warning 信息
GPIO.setmode(GPIO.BCM)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)
GPIO.output(20, 1)        # 关闭吸泵电磁阀

def pump_on():
    '''
    开启吸泵
    '''
    print('    开启吸泵')
    GPIO.output(20, 0)

def pump_off():
    '''
    关闭吸泵，吸泵放气，释放物体
    '''
    print('    关闭吸泵')
    GPIO.output(20, 1)   # 关闭吸泵电磁阀
    time.sleep(0.05)
    GPIO.output(21, 0)   # 打开泄气阀门
    time.sleep(0.2)
    GPIO.output(21, 1)
    time.sleep(0.05)
    GPIO.output(21, 0)   # 再一次泄气，确保物体释放
    time.sleep(0.2)
    GPIO.output(21, 1)
    time.sleep(0.05)
