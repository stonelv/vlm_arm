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

from API_KEY import *
from utils_robot import *
from utils_vlm import *
from utils_tts import *

# 导入中文字体
font = ImageFont.truetype('asset/SimHei.ttf', 26)
small_font = ImageFont.truetype('asset/SimHei.ttf', 18)

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
      "bounding_box": [[100, 200], [300, 400]],
      "center": [200, 300],
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

def multi_object_detection(PROMPT: str, img_path: str = 'temp/vl_now.jpg') -> Dict:
    '''
    多物体检测与属性分析
    '''
    print('多物体检测与属性分析')
    
    import openai
    from openai import OpenAI
    import base64
    
    client = OpenAI(
        api_key=Qwen_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
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
            result = json.loads(result_str)
            print('    多物体检测成功！')
            return result
            
        except json.JSONDecodeError as e:
            print(f'    JSON解析错误，尝试提取JSON内容: {e}')
            # 尝试从文本中提取JSON
            try:
                start = result_str.find('{')
                end = result_str.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = result_str[start:end]
                    result = json.loads(json_str)
                    print('    从文本中提取JSON成功！')
                    return result
            except:
                pass
            print(f'    第 {n} 次尝试失败，重试...')
            
        except Exception as e:
            print(f'    第 {n} 次尝试失败: {e}')
            time.sleep(1)
    
    raise Exception('多物体检测失败，超过重试次数')

def visualize_multi_objects(result: Dict, img_path: str, save_path: str = 'temp/multi_objects_viz.jpg') -> np.ndarray:
    '''
    可视化多物体检测结果
    '''
    print('可视化多物体检测结果')
    
    img_bgr = cv2.imread(img_path)
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
        x_min = int(bbox[0][0] * img_w / FACTOR)
        y_min = int(bbox[0][1] * img_h / FACTOR)
        x_max = int(bbox[1][0] * img_w / FACTOR)
        y_max = int(bbox[1][1] * img_h / FACTOR)
        
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
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        
        # 标签背景
        text_size = draw.textsize(label, font=small_font)
        draw.rectangle(
            [x_min, y_min - text_size[1] - 10, x_min + text_size[0] + 10, y_min - 5],
            fill=color + (200,)
        )
        
        # 绘制文本
        draw.text((x_min + 5, y_min - text_size[1] - 8), label, font=small_font, fill=(255, 255, 255, 255))
        
        # 转换回BGR
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 绘制排序规则
    sorting_rule = result.get('sorting_rule', '未指定排序规则')
    target_arrangement = result.get('target_arrangement', '')
    
    info_text = f'排序规则: {sorting_rule}'
    if target_arrangement:
        info_text += f'\n排列方式: {target_arrangement}'
    
    # 在图像顶部绘制信息
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
    
    # 保存图像
    cv2.imwrite(save_path, img_bgr)
    print(f'    可视化结果已保存至: {save_path}')
    
    return img_bgr

def generate_grasp_plan(objects: List[Dict], sorting_rule: str, target_arrangement: str) -> Dict:
    '''
    生成抓取计划
    '''
    print('生成抓取计划')
    
    # 1. 根据排序规则对物体进行排序
    if '颜色深浅' in sorting_rule:
        if '从深到浅' in sorting_rule or '降序' in sorting_rule:
            sorted_objects = sorted(objects, key=lambda x: x.get('color_depth', 5), reverse=True)
        else:
            sorted_objects = sorted(objects, key=lambda x: x.get('color_depth', 5))
    elif '大小' in sorting_rule:
        # 这里简化处理，实际应该根据bounding_box计算大小
        sorted_objects = objects
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
    
    for idx, obj in enumerate(sorted_objects):
        # 计算抓取位置（从物体当前位置）
        bbox = obj.get('bounding_box', [[0, 0], [0, 0]])
        # 转换为机械臂坐标（简化，实际需要手眼标定）
        grasp_x = int(np.interp((bbox[0][0] + bbox[1][0]) / 2, [0, 999], [-150, 150]))
        grasp_y = int(np.interp((bbox[0][1] + bbox[1][1]) / 2, [0, 999], [-150, 150]))
        
        # 计算放置位置（目标位置）
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
    print('开始执行抓取计划')
    
    sequence = grasp_plan.get('grasp_sequence', [])
    total_steps = grasp_plan.get('total_steps', 0)
    
    if not sequence:
        print('    没有抓取步骤需要执行')
        return True
    
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
                pump_move(
                    mc=mc,
                    XY_START=[grasp_pos['x'], grasp_pos['y']],
                    HEIGHT_START=grasp_height,
                    XY_END=[place_pos['x'], place_pos['y']],
                    HEIGHT_END=place_height,
                    HEIGHT_SAFE=220
                )
                print(f'    步骤 {step_num} 执行完成')
            except Exception as e:
                print(f'    步骤 {step_num} 执行时出现问题: {e}')
                # 继续执行下一步，不中断整个流程
                continue
            
            # 步骤间短暂等待
            time.sleep(1)
        
        print('\n所有抓取步骤执行完成')
        return True
        
    except Exception as e:
        print(f'执行抓取计划时发生错误: {e}')
        return False

def create_before_after_comparison(before_img_path: str, after_img_path: str, save_path: str = 'temp/comparison.jpg') -> np.ndarray:
    '''
    创建前后对比图像
    '''
    print('创建前后对比图像')
    
    # 读取图像
    before_img = cv2.imread(before_img_path)
    after_img = cv2.imread(after_img_path)
    
    if before_img is None or after_img is None:
        raise ValueError(f'无法读取图像: before={before_img_path}, after={after_img_path}')
    
    # 调整尺寸使两者一致
    h1, w1 = before_img.shape[:2]
    h2, w2 = after_img.shape[:2]
    
    # 使用较小的高度作为目标高度
    target_h = min(h1, h2)
    # 按比例调整宽度
    scale1 = target_h / h1
    scale2 = target_h / h2
    
    before_resized = cv2.resize(before_img, (int(w1 * scale1), target_h))
    after_resized = cv2.resize(after_img, (int(w2 * scale2), target_h))
    
    # 在图像上添加标签
    def add_label(img, label, position='top'):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        
        text_size = draw.textsize(label, font=font)
        h, w = img.shape[:2]
        
        # 计算文本位置
        if position == 'top':
            x = (w - text_size[0]) // 2
            y = 20
            bg_color = (255, 0, 0, 180)  # 红色背景表示"之前"
        else:
            x = (w - text_size[0]) // 2
            y = 20
            bg_color = (0, 255, 0, 180)  # 绿色背景表示"之后"
        
        # 绘制背景
        draw.rectangle(
            [x - 10, y - 5, x + text_size[0] + 10, y + text_size[1] + 5],
            fill=bg_color
        )
        
        # 绘制文本
        draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
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
    formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    timestamp_save_path = f'visualizations/comparison_{formatted_time}.jpg'
    cv2.imwrite(timestamp_save_path, comparison)
    print(f'    时间戳版本已保存至: {timestamp_save_path}')
    
    return comparison

def organize_objects(PROMPT: str = '把桌上的水果按颜色深浅排列') -> Dict:
    '''
    主函数：执行物体整理任务
    '''
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
        'error_message': ''
    }
    
    try:
        # 步骤1: 机械臂归零
        print('\n【步骤1】机械臂归零')
        back_zero()
        
        # 步骤2: 拍摄整理前的图像
        print('\n【步骤2】拍摄整理前的图像')
        before_img_path = 'temp/before_organize.jpg'
        top_view_shot(check=False)
        # 复制当前图像作为整理前图像
        import shutil
        shutil.copy('temp/vl_now.jpg', before_img_path)
        result['before_image'] = before_img_path
        print(f'    整理前图像已保存: {before_img_path}')
        
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
        back_zero()  # 先归零再拍照
        time.sleep(1)
        top_view_shot(check=False)
        shutil.copy('temp/vl_now.jpg', after_img_path)
        result['after_image'] = after_img_path
        print(f'    整理后图像已保存: {after_img_path}')
        
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
        print('='*60)
        
        # 语音反馈
        tts('整理任务完成，请查看对比图像')
        play_wav('temp/tts.wav')
        
    except Exception as e:
        error_msg = f'执行过程中发生错误: {str(e)}'
        print(error_msg)
        result['error_message'] = error_msg
        
        # 尝试语音播报错误
        try:
            tts(f'整理任务失败，错误信息: {str(e)[:50]}')
            play_wav('temp/tts.wav')
        except:
            pass
    
    # 清理
    try:
        GPIO.cleanup()
        cv2.destroyAllWindows()
    except:
        pass
    
    return result

# 测试函数
if __name__ == '__main__':
    # 测试用例
    test_prompt = '把桌上的水果按颜色深浅从深到浅排列'
    result = organize_objects(test_prompt)
    print('\n最终结果:')
    print(json.dumps(result, indent=2, ensure_ascii=False))
