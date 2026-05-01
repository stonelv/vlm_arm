import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

class ImageProcessor:
    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(os.path.join(upload_folder, 'processed'), exist_ok=True)
    
    def save_uploaded_image(self, image_data, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'image_{timestamp}.jpg'
        
        filepath = os.path.join(self.upload_folder, filename)
        
        if isinstance(image_data, str) and os.path.exists(image_data):
            img = Image.open(image_data)
            img.save(filepath)
        elif isinstance(image_data, Image.Image):
            image_data.save(filepath)
        elif isinstance(image_data, np.ndarray):
            cv2.imwrite(filepath, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
        
        return filepath
    
    def preprocess_image(self, filepath, target_size=None):
        img = Image.open(filepath)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        
        img_array = np.array(img)
        
        return img_array, img
    
    def draw_detections(self, filepath, detections, output_path=None):
        img = cv2.imread(filepath)
        img_h, img_w = img.shape[:2]
        
        for det in detections:
            name = det.get('name', 'Unknown')
            confidence = det.get('confidence', 0.0)
            bbox = det.get('bbox', None)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f'{name}: {confidence:.2f}'
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.upload_folder, 'processed', 
                                       f'detection_{timestamp}.jpg')
        
        cv2.imwrite(output_path, img)
        return output_path
    
    def get_image_info(self, filepath):
        img = Image.open(filepath)
        return {
            'width': img.width,
            'height': img.height,
            'mode': img.mode,
            'size': os.path.getsize(filepath),
            'path': filepath
        }
