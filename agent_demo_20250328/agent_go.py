# agent_go.py
# 同济子豪兄 2024-5-27
# 看懂"图像"、听懂"人话"、指哪打哪的机械臂
# 机械臂+大模型+多模态+语音识别=具身智能体Agent
# 支持任务录像与复盘功能

print('\n听得懂人话、看得懂图像、拎得清动作的具身智能机械臂！')
print('同济子豪兄 2024-5-27 \n')

import time
import traceback

from utils_asr import *
from utils_robot import *
from utils_llm import *
from utils_led import *
from utils_camera import *
from utils_robot import *
from utils_pump import *
from utils_vlm_move import *
from utils_drag_teaching import *
from utils_agent import *
from utils_tts import *
from utils_recorder import recorder, TaskStatus

print('播放欢迎词')
pump_off()
play_wav('asset/welcome.wav')

message = []
message.append({"role": "system", "content": AGENT_SYS_PROMPT})

def get_joint_angles():
    '''获取当前关节角度'''
    try:
        return mc.get_angles()
    except:
        return None

def get_coords():
    '''获取当前末端坐标'''
    try:
        return mc.get_coords()
    except:
        return None

def execute_with_recording(func_name, func_call_str):
    '''带记录的函数执行'''
    joints_before = get_joint_angles()
    coords_before = get_coords()
    
    params = {}
    try:
        if '(' in func_call_str:
            import ast
            func_part = func_call_str.split('(', 1)[0]
            args_part = func_call_str.split('(', 1)[1].rsplit(')', 1)[0]
            params['function'] = func_part
            params['args_raw'] = args_part
    except:
        pass
    
    recorder.start_action_execution(
        function_name=func_name,
        parameters=params,
        joint_angles_before=joints_before,
        coords_before=coords_before
    )
    
    status = 'success'
    error_msg = None
    ret_val = None
    
    try:
        ret_val = eval(func_call_str)
    except Exception as e:
        status = 'failed'
        error_msg = str(e)
        print(f'[Recorder] 执行错误: {func_name} - {error_msg}')
        traceback.print_exc()
    
    joints_after = get_joint_angles()
    coords_after = get_coords()
    
    recorder.end_action_execution(
        function_name=func_name,
        status=status,
        joint_angles_after=joints_after,
        coords_after=coords_after,
        error_message=error_msg,
        return_value=ret_val
    )
    
    if status == 'failed':
        raise Exception(error_msg)
    
    return ret_val

def agent_play():
    '''
    主函数，语音控制机械臂智能体编排动作
    '''
    task_id = recorder.start_task()
    task_status = TaskStatus.SUCCESS.value
    error_summary = None
    
    try:
        print('机械臂归零')
        execute_with_recording('back_zero', 'back_zero()')
        
        start_record_ok = input('是否开启录音，输入数字录音指定时长，按k打字输入，按c输入默认指令\n')
        
        order = None
        if str.isnumeric(start_record_ok):
            DURATION = int(start_record_ok)
            print(f'开始 {DURATION} 秒录音')
            record(DURATION=DURATION)
            order = speech_recognition()
            recorder.record_speech(
                audio_path='temp/speech_record.wav',
                text=order,
                duration=DURATION
            )
        elif start_record_ok == 'k':
            order = input('请输入指令')
            recorder.record_speech(
                audio_path='',
                text=order,
                duration=None
            )
        elif start_record_ok == 'c':
            order = '先归零，再摇头，然后把绿色方块放在篮球上'
            recorder.record_speech(
                audio_path='',
                text=order,
                duration=None
            )
        else:
            print('无指令，退出')
            recorder.end_task(status=TaskStatus.CANCELLED.value, error_summary='用户取消')
            raise NameError('无指令，退出')
        
        message.append({"role": "user", "content": order})
        agent_plan_output = eval(agent_plan(message))
        
        print('智能体编排动作如下\n', agent_plan_output)
        
        recorder.record_action_plan(
            original_instruction=order,
            llm_response=agent_plan_output,
            function_calls=agent_plan_output.get('function', []),
            agent_response=agent_plan_output.get('response', '')
        )
        
        plan_ok = 'c'
        if plan_ok == 'c':
            response = agent_plan_output['response']
            print('开始语音合成')
            tts(response)
            play_wav('temp/tts.wav')
            output_other = ''
            
            for each in agent_plan_output['function']:
                print('开始执行动作', each)
                func_name = each.split('(')[0] if '(' in each else each
                
                try:
                    ret = execute_with_recording(func_name, each)
                    if ret != None:
                        output_other = ret
                except Exception as e:
                    task_status = TaskStatus.FAILED.value
                    error_summary = f'执行动作失败: {each}, 错误: {str(e)}'
                    raise
        
        elif plan_ok == 'q':
            recorder.end_task(status=TaskStatus.CANCELLED.value, error_summary='用户按q退出')
            raise NameError('按q退出')
        
        agent_plan_output['response'] += '.' + output_other
        message.append({"role": "assistant", "content": str(agent_plan_output)})
        
    except Exception as e:
        if task_status != TaskStatus.CANCELLED.value:
            task_status = TaskStatus.FAILED.value
            error_summary = str(e)
        raise
    finally:
        recorder.end_task(status=task_status, error_summary=error_summary)
        print(f'\n[Recorder] 任务 {task_id} 记录完成，状态: {task_status}')

if __name__ == '__main__':
    while True:
        try:
            agent_play()
        except KeyboardInterrupt:
            print('\n用户中断')
            recorder.end_task(status=TaskStatus.CANCELLED.value, error_summary='用户键盘中断')
            break
        except Exception as e:
            print(f'\n任务执行出错: {e}')
            continue
