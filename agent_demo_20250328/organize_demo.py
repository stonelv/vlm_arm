# organize_demo.py
# 具身智能整理系统演示程序
# 支持自然语言指令、多物体检测、属性分析、抓取序列生成、前后对比图像
# 同济子豪兄 2024

print('\n' + '='*70)
print('具身智能整理系统 - 演示程序')
print('='*70)
print('功能特点：')
print('  1. 接受自然语言指令（如"按颜色深浅排列"、"叠整齐"）')
print('  2. 结合视觉模型检测物体属性（颜色、大小、姿态、易碎性等）')
print('  3. 智能生成抓取点与放置序列')
print('  4. 针对不规则/易损物品的特殊处理（轻柔操作）')
print('  5. 自动输出整理前后对比图像')
print('='*70 + '\n')

import sys
import json

# 导入模块
print('正在导入模块...')
from utils_organize import organize_objects, multi_object_detection, create_before_after_comparison
from utils_camera import check_camera
from utils_robot import back_zero

# 预设演示指令
PRESET_INSTRUCTIONS = {
    '1': '把桌上的水果按颜色深浅从深到浅排列',
    '2': '把桌上的水果按颜色深浅从浅到深排列',
    '3': '把散落的卡片叠整齐，放在桌子右侧',
    '4': '把积木按从小到大的顺序排列',
    '5': '把所有物品整理成一排，从左到右整齐摆放',
}

def show_menu():
    '''显示菜单'''
    print('\n' + '-'*50)
    print('请选择操作模式：')
    print('-'*50)
    print('1. 预设指令：按颜色深浅从深到浅排列（水果）')
    print('2. 预设指令：按颜色深浅从浅到深排列（水果）')
    print('3. 预设指令：把散落的卡片叠整齐')
    print('4. 预设指令：按大小排序（积木）')
    print('5. 预设指令：整齐排列所有物品')
    print('6. 自定义输入指令')
    print('7. 测试摄像头')
    print('8. 退出程序')
    print('-'*50)

def main():
    '''主函数'''
    
    while True:
        show_menu()
        choice = input('\n请输入选项编号（1-8）：').strip()
        
        if choice == '8':
            print('感谢使用，再见！')
            sys.exit(0)
        
        elif choice == '7':
            print('\n测试摄像头（按q键退出）')
            try:
                check_camera()
            except Exception as e:
                print(f'摄像头测试出错: {e}')
            continue
        
        elif choice == '6':
            instruction = input('\n请输入整理指令（例如：把桌上的水果按颜色深浅排列）：').strip()
            if not instruction:
                print('指令不能为空，请重新选择')
                continue
        
        elif choice in PRESET_INSTRUCTIONS:
            instruction = PRESET_INSTRUCTIONS[choice]
            print(f'\n已选择预设指令：{instruction}')
        
        else:
            print('无效选项，请重新选择')
            continue
        
        # 确认执行
        confirm = input('\n确认执行此指令？（y/n）：').strip().lower()
        if confirm != 'y':
            print('已取消执行')
            continue
        
        # 执行整理任务
        print('\n开始执行整理任务...')
        print('='*70)
        
        try:
            result = organize_objects(instruction)
            
            print('\n' + '='*70)
            print('任务执行结果摘要：')
            print('='*70)
            print(f'执行状态：{"成功" if result["success"] else "失败"}')
            print(f'原始指令：{result["prompt"]}')
            
            if result['success']:
                print(f'检测到物体数量：{len(result.get("objects_detected", []))}')
                grasp_plan = result.get('grasp_plan', {})
                print(f'抓取步骤数：{grasp_plan.get("total_steps", 0)}')
                print(f'预计耗时：{grasp_plan.get("estimated_time", 0)} 秒')
                print(f'\n生成的文件：')
                print(f'  - 整理前图像：{result["before_image"]}')
                print(f'  - 整理后图像：{result["after_image"]}')
                print(f'  - 对比图像：{result["comparison_image"]}')
                
                # 显示检测到的物体详情
                objects = result.get('objects_detected', [])
                if objects:
                    print(f'\n检测到的物体详情：')
                    for idx, obj in enumerate(objects, 1):
                        name = obj.get('name', f'物体{idx}')
                        color = obj.get('color', '未知')
                        color_depth = obj.get('color_depth', 5)
                        size = obj.get('size', '中等')
                        is_fragile = obj.get('is_fragile', False)
                        is_irregular = obj.get('is_irregular', False)
                        
                        tags = []
                        if is_fragile:
                            tags.append('易碎')
                        if is_irregular:
                            tags.append('不规则形状')
                        tag_str = f' [{", ".join(tags)}]' if tags else ''
                        
                        print(f'  {idx}. {name} - 颜色:{color}, 深浅度:{color_depth}/10, 大小:{size}{tag_str}')
            else:
                print(f'错误信息：{result.get("error_message", "未知错误")}')
            
            # 保存完整结果到JSON文件
            result_file = 'temp/last_organize_result.json'
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f'\n完整结果已保存至：{result_file}')
                
        except KeyboardInterrupt:
            print('\n\n用户中断操作')
        except Exception as e:
            print(f'\n执行过程中发生错误：{e}')
            import traceback
            traceback.print_exc()
        
        # 询问是否继续
        again = input('\n是否继续执行其他任务？（y/n）：').strip().lower()
        if again != 'y':
            print('感谢使用，再见！')
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n程序已退出')
    except Exception as e:
        print(f'\n程序发生错误：{e}')
        import traceback
        traceback.print_exc()
