import os
import json
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from datetime import datetime

from config import Config
from utils_image import ImageProcessor
from utils_vlm import VLMEngine
from utils_agent import AgentPlanner
from utils_recorder import TaskRecorder, TaskStatus

app = Flask(__name__)
app.config.from_object(Config)

image_processor = ImageProcessor(app.config['UPLOAD_FOLDER'])
vlm_engine = None
agent_planner = None
task_recorder = TaskRecorder()

def is_safe_filename(filename):
    if not filename or not isinstance(filename, str):
        return False
    if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
        return False
    if os.path.isabs(filename):
        return False
    return True

def get_safe_filepath(filename, subfolder=None):
    if not is_safe_filename(filename):
        return None
    base_folder = app.config['UPLOAD_FOLDER']
    if subfolder:
        base_folder = os.path.join(base_folder, subfolder)
    filepath = os.path.abspath(os.path.join(base_folder, filename))
    base_abs = os.path.abspath(app.config['UPLOAD_FOLDER'])
    if not filepath.startswith(base_abs + os.sep) and filepath != base_abs:
        return None
    return filepath

def init_engines():
    global vlm_engine, agent_planner
    try:
        vlm_engine = VLMEngine(app.config)
    except Exception as e:
        print(f"Warning: Failed to initialize VLM engine: {e}")
    
    agent_planner = AgentPlanner(app.config)

init_engines()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"upload_{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image_info = image_processor.get_image_info(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'image_info': image_info,
            'image_url': url_for('serve_image', filename=filename)
        })
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/uploads/<filename>')
def serve_image(filename):
    filepath = get_safe_filepath(filename)
    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    return send_file(filepath)

@app.route('/analyze', methods=['POST'])
def analyze_task():
    data = request.json
    user_instruction = data.get('instruction', '')
    filename = data.get('filename', '')
    
    if not user_instruction or not filename:
        return jsonify({'success': False, 'error': 'Missing instruction or image'})
    
    filepath = get_safe_filepath(filename)
    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'Image file not found'})
    
    task_id = task_recorder.create_task(user_instruction, filepath)
    task_recorder.update_status(TaskStatus.PLANNING)
    
    try:
        detections_result = vlm_engine.detect_objects(filepath) if vlm_engine else {
            'success': True,
            'result': {
                'detections': [
                    {
                        'name': '模拟物体A',
                        'description': '这是一个模拟检测的物体',
                        'confidence': 0.92,
                        'bbox': [100, 150, 250, 300],
                        'center': [175, 225]
                    },
                    {
                        'name': '模拟物体B',
                        'description': '另一个模拟检测的目标',
                        'confidence': 0.85,
                        'bbox': [350, 100, 500, 250],
                        'center': [425, 175]
                    }
                ],
                'summary': '检测到2个物体'
            }
        }
        
        if detections_result.get('success'):
            detections = detections_result.get('result', {}).get('detections', [])
            task_recorder.add_detections(detections, detections_result.get('raw_response'))
        else:
            return jsonify({
                'success': False,
                'error': f"Object detection failed: {detections_result.get('error')}"
            })
        
        task_analysis = vlm_engine.analyze_task(filepath, user_instruction) if vlm_engine else {
            'success': True,
            'result': {
                'task_type': 'pick_place',
                'start_object': {
                    'name': '模拟物体A',
                    'description': '需要移动的物体',
                    'bbox': [100, 150, 250, 300],
                    'center': [175, 225],
                    'confidence': 0.92
                },
                'end_object': {
                    'name': '模拟物体B',
                    'description': '目标位置',
                    'bbox': [350, 100, 500, 250],
                    'center': [425, 175],
                    'confidence': 0.85
                },
                'action_description': f'将 {user_instruction} 执行',
                'required_steps': ['识别物体', '移动到起始位置', '抓取', '移动到目标', '放置'],
                'confidence': 0.88
            }
        }
        
        if not task_analysis.get('success'):
            return jsonify({
                'success': False,
                'error': f"Task analysis failed: {task_analysis.get('error')}"
            })
        
        image_info = image_processor.get_image_info(filepath)
        execution_plan = agent_planner.generate_execution_plan(task_analysis, image_info)
        
        validation = agent_planner.validate_plan(execution_plan)
        execution_plan['validation'] = validation
        
        task_recorder.set_plan(execution_plan)
        task_recorder.update_status(TaskStatus.AWAITING_CONFIRMATION)
        
        viz_image_path = None
        if detections and len(detections) > 0:
            try:
                viz_image_path = image_processor.draw_detections(filepath, detections)
            except:
                pass
        
        response = {
            'success': True,
            'task_id': task_id,
            'task_summary': task_recorder.get_task_summary(),
            'detections': detections,
            'detections_summary': detections_result.get('result', {}).get('summary', ''),
            'task_analysis': task_analysis.get('result'),
            'execution_plan': execution_plan,
            'image_info': image_info
        }
        
        if viz_image_path:
            viz_filename = os.path.basename(viz_image_path)
            response['visualization_url'] = url_for('serve_processed_image', filename=viz_filename)
        
        task_recorder.add_system_response(f"已分析任务，生成 {execution_plan.get('total_keyframes')} 个关键帧的执行计划")
        
        return jsonify(response)
        
    except Exception as e:
        task_recorder.update_status(TaskStatus.FAILED)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/uploads/processed/<filename>')
def serve_processed_image(filename):
    filepath = get_safe_filepath(filename, subfolder='processed')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    return send_file(filepath)

@app.route('/execute', methods=['POST'])
def execute_plan():
    data = request.json
    action = data.get('action', '')
    task_id = data.get('task_id', '')
    
    if not task_recorder.current_task or task_recorder.current_task.task_id != task_id:
        return jsonify({'success': False, 'error': 'Task not found or expired'})
    
    if action == 'confirm':
        task_recorder.update_status(TaskStatus.EXECUTING)
        task_recorder.add_system_response("开始执行任务...")
        
        plan = task_recorder.current_task.plan
        keyframes = plan.get('keyframes', []) if plan else []
        
        execution_progress = []
        for i, kf in enumerate(keyframes):
            task_recorder.record_keyframe_execution(
                frame_id=kf['frame_id'],
                actual_pose=kf['pose'],
                success=True
            )
            
            execution_progress.append({
                'frame_id': kf['frame_id'],
                'type': kf['type'],
                'description': kf['description'],
                'status': 'completed'
            })
        
        task_recorder.finalize_task(TaskStatus.COMPLETED, "任务执行完成！")
        
        return jsonify({
            'success': True,
            'status': 'completed',
            'execution_progress': execution_progress,
            'task_summary': task_recorder.get_task_summary()
        })
    
    elif action == 'cancel':
        task_recorder.finalize_task(TaskStatus.CANCELLED, "任务已取消")
        return jsonify({
            'success': True,
            'status': 'cancelled',
            'message': 'Task cancelled by user'
        })
    
    elif action == 'retry':
        task_recorder.update_status(TaskStatus.PLANNING)
        return jsonify({
            'success': True,
            'status': 'retrying',
            'message': 'Ready to re-plan'
        })
    
    return jsonify({'success': False, 'error': 'Invalid action'})

@app.route('/export', methods=['POST'])
def export_task():
    data = request.json
    task_id = data.get('task_id', '')
    export_format = data.get('format', 'json')
    
    try:
        if task_recorder.current_task and task_recorder.current_task.task_id == task_id:
            export_path = task_recorder.export_task(format=export_format)
        else:
            tasks = task_recorder.list_all_tasks()
            matching = [t for t in tasks if t['task_id'] == task_id]
            if matching:
                task_recorder.load_task(matching[0]['filepath'])
                export_path = task_recorder.export_task(format=export_format)
            else:
                return jsonify({'success': False, 'error': 'Task not found'})
        
        return jsonify({
            'success': True,
            'export_path': export_path,
            'download_url': url_for('download_export', filename=os.path.basename(export_path))
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/records/<filename>')
def download_export(filename):
    if not is_safe_filename(filename):
        return jsonify({'success': False, 'error': 'Invalid filename'}), 400
    
    records_dir = os.path.abspath('records')
    filepath = os.path.abspath(os.path.join('records', filename))
    
    if not filepath.startswith(records_dir + os.sep):
        return jsonify({'success': False, 'error': 'Invalid path'}), 403
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename
    )

@app.route('/task/status', methods=['GET'])
def get_task_status():
    if task_recorder.current_task:
        return jsonify({
            'success': True,
            'task': task_recorder.get_task_summary()
        })
    return jsonify({'success': False, 'error': 'No active task'})

@app.route('/task/history', methods=['GET'])
def get_task_history():
    tasks = task_recorder.list_all_tasks()
    return jsonify({
        'success': True,
        'tasks': tasks
    })

@app.route('/visual_qa', methods=['POST'])
def visual_qa():
    data = request.json
    question = data.get('question', '')
    filename = data.get('filename', '')
    
    if not question or not filename:
        return jsonify({'success': False, 'error': 'Missing question or image'})
    
    filepath = get_safe_filepath(filename)
    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'Image file not found'})
    
    if not vlm_engine:
        return jsonify({
            'success': True,
            'answer': f"这是一个模拟回答。您的问题是：{question}。由于VLM引擎未配置，无法实际分析图片。",
            'raw_response': "模拟响应"
        })
    
    try:
        result = vlm_engine.visual_qa(filepath, question)
        if result.get('success'):
            return jsonify({
                'success': True,
                'answer': result.get('result', result.get('raw_response')),
                'raw_response': result.get('raw_response')
            })
        else:
            return jsonify({'success': False, 'error': result.get('error')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
