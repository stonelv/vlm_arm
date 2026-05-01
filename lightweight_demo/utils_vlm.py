import base64
import json
import os
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Optional, Any

class VLMEngine:
    def __init__(self, config):
        self.config = config
        self.yi_client = None
        self.qwen_client = None
        
        if config.YI_KEY:
            self.yi_client = OpenAI(
                api_key=config.YI_KEY,
                base_url="https://api.lingyiwanwu.com/v1"
            )
        
        if config.QWEN_KEY:
            self.qwen_client = OpenAI(
                api_key=config.QWEN_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
    
    def encode_image(self, image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def detect_objects(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        if prompt is None:
            prompt = "请识别图片中的所有物体，对每个物体给出：名称、描述、置信度（0-1）、边界框坐标（左上角和右下角的像素坐标）。"
        
        system_prompt = """你是一个专业的视觉检测助手。请识别图片中的所有物体，并以JSON格式输出结果。

输出格式示例：
{
  "detections": [
    {
      "name": "红色方块",
      "description": "一个红色的正方形积木块，位于图片左侧",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 400],
      "center": [200, 300]
    },
    {
      "name": "蓝色小球",
      "description": "一个蓝色的塑料小球，位于图片右侧",
      "confidence": 0.88,
      "bbox": [500, 150, 600, 250],
      "center": [550, 200]
    }
  ],
  "summary": "图片中检测到2个物体：红色方块和蓝色小球"
}

注意：
1. bbox格式为 [x1, y1, x2, y2]，分别是左上角和右下角的像素坐标
2. confidence是0到1之间的浮点数，表示检测的置信度
3. center是物体中心的像素坐标 [x, y]
4. 请确保输出是有效的JSON格式，不要包含其他内容"""
        
        return self._call_vlm_api(image_path, system_prompt + "\n\n" + prompt, "detection")
    
    def analyze_task(self, image_path: str, user_instruction: str) -> Dict[str, Any]:
        system_prompt = """你是一个机械臂任务分析助手。请根据用户的自然语言指令和图片内容，分析需要执行的操作。

请输出以下信息的JSON格式：
{
  "task_type": "move/pick_place/inspect/other",
  "start_object": {
    "name": "物体名称",
    "description": "描述",
    "bbox": [x1, y1, x2, y2],
    "center": [x, y],
    "confidence": 0.9
  },
  "end_object": {
    "name": "目标位置或物体名称",
    "description": "描述",
    "bbox": [x1, y1, x2, y2],
    "center": [x, y],
    "confidence": 0.85
  },
  "action_description": "详细描述需要执行的动作",
  "required_steps": ["步骤1", "步骤2", "步骤3"],
  "confidence": 0.92
}

注意：
1. task_type可以是：move（移动）、pick_place（抓取放置）、inspect（检查）、other（其他）
2. 如果任务不涉及物体移动，start_object和end_object可以为null
3. 请确保所有坐标都是基于图片的像素坐标"""
        
        full_prompt = f"用户指令：{user_instruction}\n\n请分析这个任务并输出JSON格式的结果。"
        
        return self._call_vlm_api(image_path, system_prompt + "\n\n" + full_prompt, "task_analysis")
    
    def _call_vlm_api(self, image_path: str, prompt: str, task_type: str) -> Dict[str, Any]:
        try:
            base64_image = self.encode_image(image_path)
            
            if self.qwen_client:
                client = self.qwen_client
                model = "qwen-vl-max-2024-11-19"
            elif self.yi_client:
                client = self.yi_client
                model = "yi-vision"
            else:
                raise ValueError("No valid VLM API key configured")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    result = json.loads(result_text[json_start:json_end])
                else:
                    result = {
                        "raw_response": result_text,
                        "task_type": task_type,
                        "error": "Failed to parse JSON response"
                    }
            
            return {
                "success": True,
                "task_type": task_type,
                "result": result,
                "raw_response": result_text,
                "model_used": model
            }
            
        except Exception as e:
            return {
                "success": False,
                "task_type": task_type,
                "error": str(e),
                "result": None
            }
    
    def visual_qa(self, image_path: str, question: str) -> Dict[str, Any]:
        system_prompt = """你是一个视觉问答助手。请根据图片内容回答用户的问题。回答要详细、准确。"""
        
        full_prompt = f"问题：{question}\n\n请根据图片内容回答。"
        
        return self._call_vlm_api(image_path, system_prompt + "\n\n" + full_prompt, "vqa")
