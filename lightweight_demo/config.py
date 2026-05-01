import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'vlm-arm-demo-secret-key'
    
    YI_KEY = os.environ.get('YI_KEY') or ''
    Qwen_KEY = os.environ.get('QWEN_KEY') or ''
    
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    VLM_MODEL = 'qwen-vl-max'
    LLM_MODEL = 'yi-large'
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    
    @staticmethod
    def init_app(app):
        pass
