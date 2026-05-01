#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs(os.path.join('uploads', 'processed'), exist_ok=True)
    os.makedirs('records', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs(os.path.join('static', 'css'), exist_ok=True)
    os.makedirs(os.path.join('static', 'js'), exist_ok=True)
    
    print("=" * 60)
    print("VLM-ARM 轻量交互Demo")
    print("=" * 60)
    print()
    print("功能说明:")
    print("  1. 上传相机图像")
    print("  2. 用自然语言下达任务")
    print("  3. 系统返回可执行计划（关键帧/位姿/夹爪开合）")
    print("  4. 可解释的中间结果（检测框、目标描述、置信度）")
    print("  5. 支持'确认后执行/取消/重试'")
    print("  6. 导出任务的完整轨迹与多模态对话记录")
    print()
    print("API配置:")
    if os.getenv('YI_KEY'):
        print("  [OK] 零一万物 API 已配置")
    else:
        print("  [WARN] 零一万物 API 未配置 (将使用模拟数据)")
    
    if os.getenv('QWEN_KEY'):
        print("  [OK] 通义千问 API 已配置")
    else:
        print("  [WARN] 通义千问 API 未配置 (将使用模拟数据)")
    print()
    print("请在浏览器中访问: http://localhost:5000")
    print("=" * 60)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
