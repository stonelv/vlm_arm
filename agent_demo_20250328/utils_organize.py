# utils_organize.py
# 具身智能整理系统
# 支持自然语言指令、多物体检测、属性分析、抓取序列生成

print('导入具身智能整理系统模块')

import time
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw
from typing import List, Dict, Tuple, Optional
import json
import os

# 全局变量
SIMULATION_MODE = False  # 模拟模式：True 不操作实际机械臂，用于测试
mc = None
pump_move = None
back_zero = None
top_view_shot = None
GPIO = None

# API Key 验证
Qwen_KEY = None
YI_KEY = None

def check_and_import_dependencies():
    '''
    检查并导入依赖
    '''
    global mc, pump_move, back_zero, top_view_shot, GPIO
    global Qwen_KEY, YI_KEY
    global SIMULATION_MODE
    
    print('正在检查依赖...')
    
    # 1. 导入 API Key
    try:
        from API_KEY import Qwen_KEY as QWEN_KEY_IMPORTED
        from API_KEY import YI_KEY as YI_KEY_IMPORTED
        Qwen_KEY = QWEN_KEY_IMPORTED
        YI_KEY = YI_KEY_IMPORTED
        
        # 检查 API Key 是否有效（非占位符）
        if Qwen_KEY and 'XXXX' not in Qwen_KEY and len(Qwen_KEY) > 10:
            print('    Qwen API Key 已配置')
        else:
            print('    警告: Qwen API Key 可能未正确配置')
        
        if YI_KEY and 'XXXX' not in YI_KEY and len(YI_KEY) > 10:
            print('    Yi API Key 已配置')
        else:
            print('    警告: Yi API Key 可能未正确配置')
            
    except ImportError as e:
        print(f'    警告: 无法导入 API_KEY: {e}')
    except Exception as e:
        print(f'    警告: API Key 导入时出错: {e}')
    
    # 2. 尝试导入机械臂相关模块
    try:
        from utils_robot import mc as MC_ROBOT
        from utils_robot import pump_move as PUMP_MOVE_FUNC
        from utils_robot import back_zero as BACK_ZERO_FUNC
        from utils_robot import top_view_shot as TOP_VIEW_SHOT_FUNC
        
        mc = MC_ROBOT
        pump_move = PUMP_MOVE_FUNC
        back_zero = BACK_ZERO_FUNC
        top_view_shot = TOP_VIEW_SHOT_FUNC
        
        # 尝试导入 GPIO
        try:
            import RPi.GPIO as GPIO_MODULE
            GPIO = GPIO_MODULE
        except:
            print('    警告: RPi.GPIO 不可用，可能不是在树莓派上运行')
        
        print('    机械臂模块导入成功')
        SIMULATION_MODE = False
        
    except Exception as e:
        print(f'    机械臂模块导入失败: {e}')
        print('    已启用模拟模式 (SIMULATION_MODE=True)')
        SIMULATION_MODE = True
        
        # 创建模拟函数
        def simulate_back_zero():
            print('    [模拟] 机械臂归零')
            time.sleep(1)
        
        def simulate_top_view_shot(check=False):
            print('    [模拟] 拍摄俯视图')
            # 创建一个模拟图像
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite('temp/vl_now.jpg', img)
            print('    [模拟] 图像已保存至 temp/vl_now.jpg')
        
        def simulate_pump_move(mc_instance, XY_START, HEIGHT_START, XY_END, HEIGHT_END, HEIGHT_SAFE=220):
            print(f'    [模拟] 吸泵移动: 从 {XY_START} 到 {XY_END}')
            print(f'    [模拟] 移动高度: 抓取={HEIGHT_START}, 放置={HEIGHT_END}, 安全={HEIGHT_SAFE}')
            time.sleep(2)
            print('    [模拟] 移动完成')
        
        back_zero = simulate_back_zero
        top_view_shot = simulate_top_view_shot
        pump_move = simulate_pump_move
    
    # 3. 导入其他工具函数
    try:
        from utils_tts import tts, play_wav
        print('    语音模块导入成功')
    except Exception as e:
        print(f'    警告: 语音模块导入失败: {e}')
        # 创建模拟函数
        global tts, play_wav
        def simulate_tts(text):
            print(f'    [模拟] 语音合成: {text[:30]}...')
        def simulate_play_wav(path):
            print(f'    [模拟] 播放音频: {path}')
        tts = simulate_tts
        play_wav = simulate_play_wav
    
    print('依赖检查完成')
    if SIMULATION_MODE:
        print('当前运行模式: 模拟模式 (无实际硬件操作)')
    else:
        print('当前运行模式: 正常模式 (连接实际硬件)')

# 执行依赖检查
check_and_import_dependencies()

# 导入中文字体
try:
    font = ImageFont.truetype('asset/SimHei.ttf', 26)
    small_font = ImageFont.truetype('asset/SimHei.ttf', 18)
except:
    print('警告: 无法加载中文字体，使用默认字体')
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

# 系统提示词：多物体检测与属性分析
SYSTEM_PROMPT_MULTI_OBJECT = '''
你是一个视觉分析助手。我会给你一张图片和一个整理指令，请帮我：
1. 识别图片中所有需要整理的物体
2. 分析每个物体的属性，包括：
   - 颜色（用RGB值或常见颜色名称描述）
   - 颜色深浅（用1-10表示，1最浅，10最深）
   - 大小（用像素边界框估计）
   - 类别（水果、卡片、积木等）
   - 姿态（是否倾斜、是否堆叠等）
   - 位置（左上角和右下角的像素坐标，归一化到0-999范围）

3. 根据整理指令，确定物体的排序规则

输出JSON格式，不要包含其他内容：
{
  "objects": [
    {
      "id": 1,
      "name": "红色苹果",
      "category": "水果",
      "color": "红色",
      "color_depth": 8,
      "size": "中等",
      "bounding_box": [[102, 505], [324, 860]],
      "center": [213, 682],
      "orientation": "正常",
      "is_fragile": true,
      "is_irregular": false
    }
  ],
  "sorting_rule": "按颜色深浅从深到浅排列",
  "target_arrangement": "从左到右依次排列"
}

注意：
- bounding_box的坐标是归一化到0-999范围的像素坐标
- 颜色深浅1最浅，10最深
- is_fragile表示是否易损（如水果、玻璃制品）
- is_irregular表示是否形状不规则
- 必须确保输出是有效的JSON格式

我现在的指令是：
'''

# 系统提示词：抓取序列生成
SYSTEM_PROMPT_GRASP_PLAN = '''
你是一个机器人运动规划助手。根据检测到的物体和整理指令，帮我生成抓取和放置序列。

已知信息：
- 机械臂使用吸泵进行抓取
- 桌面工作区域坐标范围：X从-200到200，Y从-200到200
- 安全高度：220mm
- 抓取高度：根据物体大小调整，通常80-100mm

请生成以下JSON格式的抓取计划：
{
  "grasp_sequence": [
    {
      "step": 1,
      "object_id": 1,
      "object_name": "红色苹果",
      "grasp_position": {"x": 100, "y": -50},
      "grasp_height": 90,
      "place_position": {"x": 150, "y": -100},
      "place_height": 100,
      "special_handling": "轻柔操作，降低速度",
      "reason": "这是颜色最深的水果，应放在最左边"
    }
  ],
  "total_steps": 3,
  "estimated_time": 30
}

注意：
- 对于易损物品(is_fragile=true)，需要降低移动速度，使用special_handling字段说明
- 对于不规则形状(is_irregular=true)，需要调整抓取点到物体中心
- 放置位置要考虑物体大小，避免重叠
- 抓取序列要考虑移动效率，减少机械臂移动距离

我现在的物体信息和整理指令是：
'''

def check_api_key() -> bool:
    '''
    检查 API Key 是否可用
    '''
    global Qwen_KEY
    
    if Qwen_KEY is None:
        print('错误: API Key 未初始化')
        return False
    
    if 'XXXX' in Qwen_KEY or len(Qwen_KEY) < 10:
        print('错误: API Key 看起来是占位符，请在 API_KEY.py 中配置正确的 Key')
        print(f'当前 Key: {Qwen_KEY[:20]}...')
        return False
    
    return True

def multi_object_detection(PROMPT: str, img_path: str = 'temp/vl_now.jpg') -> Dict:
    '''
    多物体检测与属性分析
    '''
    print('多物体检测与属性分析')
    
    # 检查 API Key
    if not check_api_key():
        # 如果 API Key 不可用，在模拟模式下返回模拟数据
        if SIMULATION_MODE:
            print('    [模拟模式] 返回模拟检测数据')
            return get_simulation_detection_data(PROMPT)
        else:
            raise Exception('API Key 未正确配置，无法调用视觉模型')
    
    import openai
    from openai import OpenAI
    import base64
    
    client = OpenAI(
        api_key=Qwen_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 检查图像文件是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'图像文件不存在: {img_path}')
    
    with open(img_path, 'rb') as image_file:
        image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')
    
    full_prompt = SYSTEM_PROMPT_MULTI_OBJECT + PROMPT
    
    for n in range(1, 4):
        try:
            print(f'    尝试第 {n} 次访问多模态大模型')
            completion = client.chat.completions.create(
                model="qwen-vl-max-2024-11-19",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {"type": "image_url", "image_url": {"url": image}}
                        ]
                    }
                ]
            )
            
            result_str = completion.choices[0].message.content.strip()
            print('    大模型返回结果:')
            print(result_str)
            
            # 尝试解析JSON
            try:
                result = json.loads(result_str)
                print('    多物体检测成功！')
                return result
            except json.JSONDecodeError:
                print('    JSON解析错误，尝试从文本中提取JSON...')
                # 尝试从文本中提取JSON
                start = result_str.find('{')
                end = result_str.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = result_str[start:end]
                    result = json.loads(json_str)
                    print('    从文本中提取JSON成功！')
                    return result
                raise
                
        except json.JSONDecodeError as e:
            print(f'    JSON解析错误: {e}')
            if n < 3:
                print(f'    第 {n} 次尝试失败，重试...')
                time.sleep(1)
            else:
                # 最后一次尝试失败，如果是模拟模式，返回模拟数据
                if SIMULATION_MODE:
                    print('    [模拟模式] 返回模拟检测数据')
                    return get_simulation_detection_data(PROMPT)
                raise Exception(f'多物体检测失败，JSON解析错误: {e}')
            
        except Exception as e:
            print(f'    第 {n} 次尝试失败: {e}')
            if n < 3:
                time.sleep(1)
            else:
                if SIMULATION_MODE:
                    print('    [模拟模式] 返回模拟检测数据')
                    return get_simulation_detection_data(PROMPT)
                raise Exception(f'多物体检测失败，超过重试次数: {e}')
    
    raise Exception('多物体检测失败，超过重试次数')

def get_simulation_detection_data(prompt: str) -> Dict:
    '''
    返回模拟的检测数据（用于测试）
    '''
    # 根据不同的提示词返回不同的模拟数据
    if '水果' in prompt:
        return {
            "objects": [
                {
                    "id": 1,
                    "name": "红苹果",
                    "category": "水果",
                    "color": "红色",
                    "color_depth": 8,
                    "size": "中等",
                    "bounding_box": [[150, 200], [350, 400]],
                    "center": [250, 300],
                    "orientation": "正常",
                    "is_fragile": True,
                    "is_irregular": False
                },
                {
                    "id": 2,
                    "name": "黄香蕉",
                    "category": "水果",
                    "color": "黄色",
                    "color_depth": 4,
                    "size": "中等",
                    "bounding_box": [[400, 150], [600, 350]],
                    "center": [500, 250],
                    "orientation": "倾斜",
                    "is_fragile": True,
                    "is_irregular": True
                },
                {
                    "id": 3,
                    "name": "绿葡萄",
                    "category": "水果",
                    "color": "绿色",
                    "color_depth": 6,
                    "size": "小",
                    "bounding_box": [[200, 400], [400, 550]],
                    "center": [300, 475],
                    "orientation": "正常",
                    "is_fragile": True,
                    "is_irregular": False
                }
            ],
            "sorting_rule": "按颜色深浅从深到浅排列",
            "target_arrangement": "从左到右依次排列"
        }
    elif '卡片' in prompt or '叠' in prompt:
        return {
            "objects": [
                {
                    "id": 1,
                    "name": "蓝色卡片",
                    "category": "卡片",
                    "color": "蓝色",
                    "color_depth": 7,
                    "size": "中等",
                    "bounding_box": [[100, 100], [250, 200]],
                    "center": [175, 150],
                    "orientation": "倾斜",
                    "is_fragile": False,
                    "is_irregular": False
                },
                {
                    "id": 2,
                    "name": "红色卡片",
                    "category": "卡片",
                    "color": "红色",
                    "color_depth": 8,
                    "size": "中等",
                    "bounding_box": [[300, 200], [450, 300]],
                    "center": [375, 250],
                    "orientation": "正常",
                    "is_fragile": False,
                    "is_irregular": False
                }
            ],
            "sorting_rule": "叠放整齐",
            "target_arrangement": "在桌子右侧堆叠"
        }
    else:
        # 默认返回积木数据
        return {
            "objects": [
                {
                    "id": 1,
                    "name": "蓝色积木",
                    "category": "积木",
                    "color": "蓝色",
                    "color_depth": 7,
                    "size": "大",
                    "bounding_box": [[100, 200], [300, 400]],
                    "center": [200, 300],
                    "orientation": "正常",
                    "is_fragile": False,
                    "is_irregular": False
                },
                {
                    "id": 2,
                    "name": "红色积木",
                    "category": "积木",
                    "color": "红色",
                    "color_depth": 8,
                    "size": "中等",
                    "bounding_box": [[350, 250], [500, 400]],
                    "center": [425, 325],
                    "orientation": "正常",
                    "is_fragile": False,
                    "is_irregular": False
                }
            ],
            "sorting_rule": "按大小排序",
            "target_arrangement": "从左到右排列"
        }

def visualize_multi_objects(result: Dict, img_path: str, save_path: str = 'temp/multi_objects_viz.jpg') -> np.ndarray:
    '''
    可视化多物体检测结果
    '''
    print('可视化多物体检测结果')
    
    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f'    警告: 图像文件不存在 {img_path}，创建空白图像')
        # 创建一个空白图像
        img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        img_bgr[:] = (255, 255, 255)  # 白色背景
    else:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f'    警告: 无法读取图像 {img_path}，创建空白图像')
            img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            img_bgr[:] = (255, 255, 255)
    
    img_h = img_bgr.shape[0]
    img_w = img_bgr.shape[1]
    FACTOR = 999
    
    # 生成颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    
    objects = result.get('objects', [])
    
    for idx, obj in enumerate(objects):
        # 解析边界框
        bbox = obj.get('bounding_box', [[0, 0], [0, 0]])
        try:
            x_min = int(bbox[0][0] * img_w / FACTOR)
            y_min = int(bbox[0][1] * img_h / FACTOR)
            x_max = int(bbox[1][0] * img_w / FACTOR)
            y_max = int(bbox[1][1] * img_h / FACTOR)
        except:
            # 如果坐标解析失败，使用默认位置
            x_min = 50 + idx * 150
            y_min = 50
            x_max = x_min + 100
            y_max = y_min + 100
        
        # 确保坐标在图像范围内
        x_min = max(0, min(x_min, img_w - 1))
        y_min = max(0, min(y_min, img_h - 1))
        x_max = max(x_min + 1, min(x_max, img_w))
        y_max = max(y_min + 1, min(y_max, img_h))
        
        # 计算中心点
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        
        # 选择颜色
        color = colors[idx % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), color, thickness=3)
        
        # 绘制中心点
        cv2.circle(img_bgr, (center_x, center_y), 8, color, thickness=-1)
        
        # 准备标签
        name = obj.get('name', f'物体{idx+1}')
        color_depth = obj.get('color_depth', 5)
        is_fragile = obj.get('is_fragile', False)
        is_irregular = obj.get('is_irregular', False)
        
        label = f'{name} (深浅:{color_depth})'
        if is_fragile:
            label += ' [易碎]'
        if is_irregular:
            label += ' [不规则]'
        
        # 绘制标签
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            
            # 标签背景
            text_size = draw.textsize(label, font=small_font)
            bg_x = max(0, x_min)
            bg_y = max(0, y_min - text_size[1] - 10)
            draw.rectangle(
                [bg_x, bg_y, bg_x + text_size[0] + 10, bg_y + text_size[1] + 5],
                fill=color + (200,)
            )
            
            # 绘制文本
            text_x = bg_x + 5
            text_y = bg_y + 2
            draw.text((text_x, text_y), label, font=small_font, fill=(255, 255, 255, 255))
            
            # 转换回BGR
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f'    警告: 绘制标签失败: {e}')
    
    # 绘制排序规则
    sorting_rule = result.get('sorting_rule', '未指定排序规则')
    target_arrangement = result.get('target_arrangement', '')
    
    info_text = f'排序规则: {sorting_rule}'
    if target_arrangement:
        info_text += f'\n排列方式: {target_arrangement}'
    
    # 在图像顶部绘制信息
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        
        lines = info_text.split('\n')
        y_offset = 20
        for line in lines:
            text_size = draw.textsize(line, font=font)
            # 背景
            draw.rectangle(
                [10, y_offset - 5, 20 + text_size[0], y_offset + text_size[1] + 5],
                fill=(0, 0, 0, 180)
            )
            # 文本
            draw.text((15, y_offset), line, font=font, fill=(255, 255, 255, 255))
            y_offset += text_size[1] + 10
        
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f'    警告: 绘制信息失败: {e}')
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # 保存图像
    cv2.imwrite(save_path, img_bgr)
    print(f'    可视化结果已保存至: {save_path}')
    
    return img_bgr

def generate_grasp_plan(objects: List[Dict], sorting_rule: str, target_arrangement: str) -> Dict:
    '''
    生成抓取计划
    '''
    print('生成抓取计划')
    
    if not objects:
        print('    警告: 没有物体需要排序')
        return {"grasp_sequence": [], "total_steps": 0, "estimated_time": 0}
    
    # 1. 根据排序规则对物体进行排序
    sorted_objects = objects
    
    if '颜色深浅' in sorting_rule:
        if '从深到浅' in sorting_rule or '降序' in sorting_rule:
            sorted_objects = sorted(objects, key=lambda x: x.get('color_depth', 5), reverse=True)
        else:
            sorted_objects = sorted(objects, key=lambda x: x.get('color_depth', 5))
    elif '大小' in sorting_rule:
        # 根据尺寸排序
        def get_size_order(obj):
            size = obj.get('size', '中等')
            if size == '大' or size == 'large':
                return 3
            elif size == '小' or size == 'small':
                return 1
            else:
                return 2
        sorted_objects = sorted(objects, key=get_size_order)
    else:
        # 默认按id顺序
        sorted_objects = sorted(objects, key=lambda x: x.get('id', 0))
    
    print(f'    排序后物体顺序: {[obj.get("name", "") for obj in sorted_objects]}')
    
    # 2. 生成放置位置
    # 计算目标放置区域的位置
    # 这里简化为从左到右线性排列
    grasp_sequence = []
    
    # 基础位置参数
    base_x = -100  # 起始X坐标
    base_y = -100  # 起始Y坐标
    step_x = 80    # 每个物体的X间距
    step_y = 0     # Y方向间距（如果是多行排列）
    
    # 如果是叠放，调整放置位置
    if '叠' in target_arrangement or '堆叠' in sorting_rule:
        step_x = 0  # 叠放时X不变化
        base_y = -100
    
    for idx, obj in enumerate(sorted_objects):
        # 计算抓取位置（从物体当前位置）
        bbox = obj.get('bounding_box', [[0, 0], [0, 0]])
        # 转换为机械臂坐标（简化，实际需要手眼标定）
        try:
            center_x = (bbox[0][0] + bbox[1][0]) / 2
            center_y = (bbox[0][1] + bbox[1][1]) / 2
            grasp_x = int(np.interp(center_x, [0, 999], [-150, 150]))
            grasp_y = int(np.interp(center_y, [0, 999], [-150, 150]))
        except:
            # 如果坐标解析失败，使用默认位置
            grasp_x = -50 + idx * 50
            grasp_y = -50
        
        # 计算放置位置（目标位置）
        # 如果是叠放，所有物体放在同一个位置
        if '叠' in target_arrangement or '堆叠' in sorting_rule:
            place_x = base_x  # 同一位置
            place_y = base_y
        else:
            place_x = base_x + idx * step_x
            place_y = base_y + idx * step_y
        
        # 确定特殊处理方式
        special_handling = []
        if obj.get('is_fragile', False):
            special_handling.append('易碎物品，降低移动速度至50%')
        if obj.get('is_irregular', False):
            special_handling.append('不规则形状，确保抓取点在中心')
        
        # 确定抓取高度
        size = obj.get('size', '中等')
        if size == '大' or size == 'large':
            grasp_height = 80
        elif size == '小' or size == 'small':
            grasp_height = 100
        else:
            grasp_height = 90
        
        grasp_step = {
            "step": idx + 1,
            "object_id": obj.get('id', idx + 1),
            "object_name": obj.get('name', f'物体{idx+1}'),
            "grasp_position": {"x": grasp_x, "y": grasp_y},
            "grasp_height": grasp_height,
            "place_position": {"x": place_x, "y": place_y},
            "place_height": 100,
            "special_handling": "；".join(special_handling) if special_handling else "无特殊处理",
            "reason": f"排序第{idx+1}位，放置在目标位置{idx+1}"
        }
        
        grasp_sequence.append(grasp_step)
    
    grasp_plan = {
        "grasp_sequence": grasp_sequence,
        "total_steps": len(grasp_sequence),
        "estimated_time": len(grasp_sequence) * 15  # 每个步骤约15秒
    }
    
    print(f'    生成抓取计划完成，共 {grasp_plan["total_steps"]} 个步骤')
    
    return grasp_plan

def execute_grasp_plan(grasp_plan: Dict) -> bool:
    '''
    执行抓取计划
    '''
    global pump_move, mc, SIMULATION_MODE
    
    print('开始执行抓取计划')
    
    sequence = grasp_plan.get('grasp_sequence', [])
    total_steps = grasp_plan.get('total_steps', 0)
    
    if not sequence:
        print('    没有抓取步骤需要执行')
        return True
    
    # 检查必要的函数是否可用
    if pump_move is None:
        print('    错误: pump_move 函数未定义')
        if SIMULATION_MODE:
            print('    [模拟模式] 继续执行模拟操作')
        else:
            return False
    
    try:
        for idx, step in enumerate(sequence):
            step_num = step.get('step', idx + 1)
            obj_name = step.get('object_name', '未知物体')
            print(f'\n    === 执行步骤 {step_num}/{total_steps}: {obj_name} ===')
            
            # 获取位置信息
            grasp_pos = step.get('grasp_position', {'x': 0, 'y': 0})
            grasp_height = step.get('grasp_height', 90)
            place_pos = step.get('place_position', {'x': 0, 'y': 0})
            place_height = step.get('place_height', 100)
            special_handling = step.get('special_handling', '')
            
            print(f'    抓取位置: ({grasp_pos["x"]}, {grasp_pos["y"]}), 高度: {grasp_height}')
            print(f'    放置位置: ({place_pos["x"]}, {place_pos["y"]}), 高度: {place_height}')
            print(f'    特殊处理: {special_handling}')
            
            # 确定速度（对于易碎物品降低速度）
            speed = 20
            if '易碎' in special_handling or 'fragile' in special_handling.lower():
                speed = 10
                print(f'    易碎物品，使用较低速度: {speed}')
            
            # 执行抓取移动
            try:
                if pump_move is not None:
                    if SIMULATION_MODE:
                        print(f'    [模拟] 调用 pump_move 函数')
                    
                    pump_move(
                        mc=mc,
                        XY_START=[grasp_pos['x'], grasp_pos['y']],
                        HEIGHT_START=grasp_height,
                        XY_END=[place_pos['x'], place_pos['y']],
                        HEIGHT_END=place_height,
                        HEIGHT_SAFE=220
                    )
                    print(f'    步骤 {step_num} 执行完成')
                else:
                    print(f'    警告: pump_move 不可用，跳过此步骤')
                    
            except Exception as e:
                print(f'    步骤 {step_num} 执行时出现问题: {e}')
                import traceback
                traceback.print_exc()
                # 继续执行下一步，不中断整个流程
                continue
            
            # 步骤间短暂等待
            time.sleep(1)
        
        print('\n所有抓取步骤执行完成')
        return True
        
    except Exception as e:
        print(f'执行抓取计划时发生错误: {e}')
        import traceback
        traceback.print_exc()
        return False

def create_before_after_comparison(before_img_path: str, after_img_path: str, save_path: str = 'temp/comparison.jpg') -> np.ndarray:
    '''
    创建前后对比图像
    '''
    print('创建前后对比图像')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # 读取图像，如果不存在则创建空白图像
    def load_or_create(img_path):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img
        print(f'    警告: 无法读取图像 {img_path}，创建空白图像')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (200, 200, 200)  # 灰色背景
        return img
    
    before_img = load_or_create(before_img_path)
    after_img = load_or_create(after_img_path)
    
    # 调整尺寸使两者一致
    h1, w1 = before_img.shape[:2]
    h2, w2 = after_img.shape[:2]
    
    # 使用较小的高度作为目标高度
    target_h = min(h1, h2, 480)  # 最大高度480
    # 按比例调整宽度
    scale1 = target_h / h1
    scale2 = target_h / h2
    
    before_resized = cv2.resize(before_img, (int(w1 * scale1), target_h))
    after_resized = cv2.resize(after_img, (int(w2 * scale2), target_h))
    
    # 在图像上添加标签
    def add_label(img, label, position='top'):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            
            text_size = draw.textsize(label, font=font)
            h, w = img.shape[:2]
            
            # 计算文本位置
            x = (w - text_size[0]) // 2
            y = 20
            
            # 选择颜色
            if position == 'top':
                bg_color = (255, 0, 0, 180)  # 红色背景表示"之前"
            else:
                bg_color = (0, 255, 0, 180)  # 绿色背景表示"之后"
            
            # 绘制背景
            draw.rectangle(
                [x - 10, y - 5, x + text_size[0] + 10, y + text_size[1] + 5],
                fill=bg_color
            )
            
            # 绘制文本
            draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
            
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f'    警告: 添加标签失败: {e}')
            return img
    
    # 添加标签
    before_labeled = add_label(before_resized, '整理前', 'top')
    after_labeled = add_label(after_resized, '整理后', 'bottom')
    
    # 水平拼接
    comparison = np.hstack([before_labeled, after_labeled])
    
    # 添加分隔线
    h, w = comparison.shape[:2]
    mid = before_labeled.shape[1]
    cv2.line(comparison, (mid, 0), (mid, h), (255, 255, 255), thickness=3)
    
    # 保存对比图
    cv2.imwrite(save_path, comparison)
    print(f'    前后对比图已保存至: {save_path}')
    
    # 同时保存带时间戳的版本
    try:
        formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        vis_dir = 'visualizations'
        os.makedirs(vis_dir, exist_ok=True)
        timestamp_save_path = f'{vis_dir}/comparison_{formatted_time}.jpg'
        cv2.imwrite(timestamp_save_path, comparison)
        print(f'    时间戳版本已保存至: {timestamp_save_path}')
    except Exception as e:
        print(f'    警告: 保存时间戳版本失败: {e}')
    
    return comparison

def organize_objects(PROMPT: str = '把桌上的水果按颜色深浅排列') -> Dict:
    '''
    主函数：执行物体整理任务
    '''
    global back_zero, top_view_shot, SIMULATION_MODE
    global tts, play_wav
    
    print('\n' + '='*60)
    print('具身智能整理系统启动')
    print('='*60)
    print(f'任务指令: {PROMPT}')
    
    result = {
        'success': False,
        'prompt': PROMPT,
        'before_image': '',
        'after_image': '',
        'comparison_image': '',
        'objects_detected': [],
        'grasp_plan': {},
        'error_message': '',
        'simulation_mode': SIMULATION_MODE
    }
    
    try:
        # 确保 temp 目录存在
        os.makedirs('temp', exist_ok=True)
        
        # 步骤1: 机械臂归零
        print('\n【步骤1】机械臂归零')
        if back_zero:
            back_zero()
        else:
            print('    警告: back_zero 函数不可用')
        
        # 步骤2: 拍摄整理前的图像
        print('\n【步骤2】拍摄整理前的图像')
        before_img_path = 'temp/before_organize.jpg'
        
        if top_view_shot:
            top_view_shot(check=False)
            # 复制当前图像作为整理前图像
            import shutil
            if os.path.exists('temp/vl_now.jpg'):
                shutil.copy('temp/vl_now.jpg', before_img_path)
                result['before_image'] = before_img_path
                print(f'    整理前图像已保存: {before_img_path}')
            else:
                print('    警告: vl_now.jpg 不存在')
                # 创建一个空白图像
                blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
                blank_img[:] = (255, 255, 255)
                cv2.imwrite(before_img_path, blank_img)
                result['before_image'] = before_img_path
        else:
            print('    警告: top_view_shot 函数不可用，创建空白图像')
            blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
            blank_img[:] = (255, 255, 255)
            cv2.imwrite(before_img_path, blank_img)
            result['before_image'] = before_img_path
        
        # 步骤3: 多物体检测与属性分析
        print('\n【步骤3】多物体检测与属性分析')
        detection_result = multi_object_detection(PROMPT, img_path=before_img_path)
        result['objects_detected'] = detection_result.get('objects', [])
        
        # 可视化检测结果
        visualize_multi_objects(detection_result, before_img_path, 'temp/detection_viz.jpg')
        
        # 步骤4: 生成抓取计划
        print('\n【步骤4】生成抓取计划')
        objects = detection_result.get('objects', [])
        sorting_rule = detection_result.get('sorting_rule', PROMPT)
        target_arrangement = detection_result.get('target_arrangement', '从左到右排列')
        
        if not objects:
            print('警告: 未检测到任何物体，任务结束')
            result['error_message'] = '未检测到任何物体'
            return result
        
        grasp_plan = generate_grasp_plan(objects, sorting_rule, target_arrangement)
        result['grasp_plan'] = grasp_plan
        
        # 步骤5: 执行抓取计划
        print('\n【步骤5】执行抓取计划')
        execute_success = execute_grasp_plan(grasp_plan)
        
        # 步骤6: 拍摄整理后的图像
        print('\n【步骤6】拍摄整理后的图像')
        after_img_path = 'temp/after_organize.jpg'
        
        if back_zero:
            back_zero()  # 先归零再拍照
        time.sleep(1)
        
        if top_view_shot:
            top_view_shot(check=False)
            import shutil
            if os.path.exists('temp/vl_now.jpg'):
                shutil.copy('temp/vl_now.jpg', after_img_path)
                result['after_image'] = after_img_path
                print(f'    整理后图像已保存: {after_img_path}')
            else:
                print('    警告: vl_now.jpg 不存在，创建空白图像')
                # 在模拟模式下，创建一个修改过的图像表示整理后
                if SIMULATION_MODE:
                    print('    [模拟模式] 创建模拟的整理后图像')
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    img[:] = (200, 255, 200)  # 浅绿色表示已整理
                    # 绘制一些整齐排列的矩形表示整理好的物体
                    for i in range(min(3, len(objects))):
                        x = 100 + i * 150
                        cv2.rectangle(img, (x, 200), (x + 100, 300), (0, 100 + i * 50, 200), -1)
                    cv2.imwrite(after_img_path, img)
                else:
                    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    blank_img[:] = (255, 255, 255)
                    cv2.imwrite(after_img_path, blank_img)
                result['after_image'] = after_img_path
        else:
            print('    警告: top_view_shot 函数不可用')
            blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
            blank_img[:] = (255, 255, 255)
            cv2.imwrite(after_img_path, blank_img)
            result['after_image'] = after_img_path
        
        # 步骤7: 创建前后对比图
        print('\n【步骤7】创建前后对比图像')
        comparison_path = 'temp/comparison.jpg'
        create_before_after_comparison(before_img_path, after_img_path, comparison_path)
        result['comparison_image'] = comparison_path
        
        # 任务完成
        result['success'] = True
        print('\n' + '='*60)
        print('整理任务完成！')
        print(f'整理前图像: {result["before_image"]}')
        print(f'整理后图像: {result["after_image"]}')
        print(f'对比图像: {result["comparison_image"]}')
        if SIMULATION_MODE:
            print('运行模式: 模拟模式 (无实际硬件操作)')
        print('='*60)
        
        # 语音反馈
        try:
            if tts and play_wav:
                tts('整理任务完成，请查看对比图像')
                play_wav('temp/tts.wav')
        except Exception as e:
            print(f'    警告: 语音反馈失败: {e}')
        
    except Exception as e:
        error_msg = f'执行过程中发生错误: {str(e)}'
        print(error_msg)
        import traceback
        traceback.print_exc()
        result['error_message'] = error_msg
        
        # 尝试语音播报错误
        try:
            if tts and play_wav:
                tts(f'整理任务失败，错误信息: {str(e)[:50]}')
                play_wav('temp/tts.wav')
        except:
            pass
    
    # 清理
    try:
        global GPIO
        if GPIO and not SIMULATION_MODE:
            GPIO.cleanup()
        cv2.destroyAllWindows()
    except:
        pass
    
    # 保存完整结果到JSON文件
    try:
        result_file = 'temp/last_organize_result.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'\n完整结果已保存至: {result_file}')
    except Exception as e:
        print(f'警告: 保存结果JSON失败: {e}')
    
    return result

# 测试函数
if __name__ == '__main__':
    # 测试用例
    test_prompt = '把桌上的水果按颜色深浅从深到浅排列'
    result = organize_objects(test_prompt)
    print('\n最终结果:')
    print(json.dumps(result, indent=2, ensure_ascii=False))
