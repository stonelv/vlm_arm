# utils_calibration.py
# 视觉坐标标定与点击取点工具
# 支持多标定点仿射变换、数据保存加载、误差评估

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict


@dataclass
class CalibrationPoint:
    pixel_x: int
    pixel_y: int
    robot_x: float
    robot_y: float
    label: str = ""


@dataclass
class CalibrationResult:
    transform_matrix: Optional[np.ndarray]
    affine_matrix: Optional[np.ndarray]
    points: List[CalibrationPoint]
    errors: Dict[str, float]
    is_valid: bool = False


class Calibrator:
    def __init__(self):
        self.calibration_points: List[CalibrationPoint] = []
        self.transform_matrix: Optional[np.ndarray] = None
        self.affine_matrix: Optional[np.ndarray] = None
        self.calibration_file = "calibration_data.json"

    def add_calibration_point(
        self, 
        pixel_coord: Tuple[int, int], 
        robot_coord: Tuple[float, float],
        label: str = ""
    ) -> None:
        point = CalibrationPoint(
            pixel_x=pixel_coord[0],
            pixel_y=pixel_coord[1],
            robot_x=robot_coord[0],
            robot_y=robot_coord[1],
            label=label
        )
        self.calibration_points.append(point)
        print(f"已添加标定点: 像素{point.pixel_x, point.pixel_y} -> 机械臂{point.robot_x, point.robot_y}")

    def remove_last_point(self) -> None:
        if self.calibration_points:
            removed = self.calibration_points.pop()
            print(f"已移除最后一个标定点: 像素{removed.pixel_x, removed.pixel_y}")
        else:
            print("没有标定点可移除")

    def clear_points(self) -> None:
        self.calibration_points.clear()
        self.transform_matrix = None
        self.affine_matrix = None
        print("已清除所有标定点")

    def get_points(self) -> List[CalibrationPoint]:
        return self.calibration_points

    def calculate_linear_transform(self) -> bool:
        if len(self.calibration_points) < 2:
            print("至少需要2个标定点进行线性标定")
            return False
        
        if len(self.calibration_points) == 2:
            p1, p2 = self.calibration_points
            pixel_points = np.array([
                [p1.pixel_x, p1.pixel_y],
                [p2.pixel_x, p2.pixel_y]
            ], dtype=np.float32)
            robot_points = np.array([
                [p1.robot_x, p1.robot_y],
                [p2.robot_x, p2.robot_y]
            ], dtype=np.float32)
        else:
            pixel_points = np.array([
                [p.pixel_x, p.pixel_y] for p in self.calibration_points
            ], dtype=np.float32)
            robot_points = np.array([
                [p.robot_x, p.robot_y] for p in self.calibration_points
            ], dtype=np.float32)
        
        self.transform_matrix = cv2.estimateAffine2D(
            pixel_points, robot_points
        )[0]
        
        if self.transform_matrix is None:
            print("线性变换矩阵计算失败")
            return False
        
        self.affine_matrix = self.transform_matrix.copy()
        print(f"线性标定完成，使用 {len(self.calibration_points)} 个标定点")
        return True

    def pixel_to_robot(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        if self.transform_matrix is None:
            if len(self.calibration_points) == 2:
                return self._pixel_to_robot_linear_interpolation(pixel_x, pixel_y)
            else:
                raise ValueError("标定未完成，请先进行标定")
        
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        robot_point = cv2.transform(pixel_point, self.transform_matrix)
        return float(robot_point[0][0][0]), float(robot_point[0][0][1])

    def _pixel_to_robot_linear_interpolation(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        if len(self.calibration_points) != 2:
            raise ValueError("需要恰好2个标定点进行线性插值")
        
        p1, p2 = self.calibration_points
        
        X_cali_im = [p1.pixel_x, p2.pixel_x]
        X_cali_mc = [p1.robot_x, p2.robot_x]
        
        Y_cali_im = [p2.pixel_y, p1.pixel_y]
        Y_cali_mc = [p2.robot_y, p1.robot_y]
        
        X_mc = np.interp(pixel_x, X_cali_im, X_cali_mc)
        Y_mc = np.interp(pixel_y, Y_cali_im, Y_cali_mc)
        
        return float(X_mc), float(Y_mc)

    def robot_to_pixel(self, robot_x: float, robot_y: float) -> Tuple[int, int]:
        if self.transform_matrix is None:
            raise ValueError("标定未完成，请先进行标定")
        
        inverse_matrix = cv2.invertAffineTransform(self.transform_matrix)
        robot_point = np.array([[[robot_x, robot_y]]], dtype=np.float32)
        pixel_point = cv2.transform(robot_point, inverse_matrix)
        return int(pixel_point[0][0][0]), int(pixel_point[0][0][1])

    def evaluate_errors(self) -> Dict[str, float]:
        if len(self.calibration_points) < 2:
            return {"mean_error": -1, "max_error": -1, "min_error": -1, "std_error": -1}
        
        errors = []
        for point in self.calibration_points:
            try:
                pred_robot_x, pred_robot_y = self.pixel_to_robot(point.pixel_x, point.pixel_y)
                error = np.sqrt(
                    (pred_robot_x - point.robot_x) ** 2 + 
                    (pred_robot_y - point.robot_y) ** 2
                )
                errors.append(error)
            except:
                continue
        
        if not errors:
            return {"mean_error": -1, "max_error": -1, "min_error": -1, "std_error": -1}
        
        errors = np.array(errors)
        return {
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "min_error": float(np.min(errors)),
            "std_error": float(np.std(errors))
        }

    def save_calibration(self, filepath: Optional[str] = None) -> bool:
        if filepath is None:
            filepath = self.calibration_file
        
        data = {
            "points": [asdict(p) for p in self.calibration_points],
            "transform_matrix": self.transform_matrix.tolist() if self.transform_matrix is not None else None,
            "affine_matrix": self.affine_matrix.tolist() if self.affine_matrix is not None else None
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"标定数据已保存至: {filepath}")
            return True
        except Exception as e:
            print(f"保存标定数据失败: {e}")
            return False

    def load_calibration(self, filepath: Optional[str] = None) -> bool:
        if filepath is None:
            filepath = self.calibration_file
        
        if not os.path.exists(filepath):
            print(f"标定文件不存在: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.calibration_points = []
            for p in data.get("points", []):
                point = CalibrationPoint(
                    pixel_x=p["pixel_x"],
                    pixel_y=p["pixel_y"],
                    robot_x=p["robot_x"],
                    robot_y=p["robot_y"],
                    label=p.get("label", "")
                )
                self.calibration_points.append(point)
            
            tm = data.get("transform_matrix")
            if tm is not None:
                self.transform_matrix = np.array(tm, dtype=np.float32)
            
            am = data.get("affine_matrix")
            if am is not None:
                self.affine_matrix = np.array(am, dtype=np.float32)
            
            print(f"已加载 {len(self.calibration_points)} 个标定点")
            return True
        except Exception as e:
            print(f"加载标定数据失败: {e}")
            return False

    def is_calibrated(self) -> bool:
        return self.transform_matrix is not None or len(self.calibration_points) >= 2

    def get_calibration_info(self) -> str:
        info = ["=" * 50]
        info.append("标定信息")
        info.append("=" * 50)
        info.append(f"标定点数量: {len(self.calibration_points)}")
        
        for i, p in enumerate(self.calibration_points, 1):
            label = f" [{p.label}]" if p.label else ""
            info.append(f"  标定点{i}{label}: 像素({p.pixel_x}, {p.pixel_y}) -> 机械臂({p.robot_x:.1f}, {p.robot_y:.1f})")
        
        if self.is_calibrated():
            errors = self.evaluate_errors()
            if errors["mean_error"] >= 0:
                info.append(f"\n误差评估:")
                info.append(f"  平均误差: {errors['mean_error']:.2f} mm")
                info.append(f"  最大误差: {errors['max_error']:.2f} mm")
                info.append(f"  最小误差: {errors['min_error']:.2f} mm")
                info.append(f"  标准差:   {errors['std_error']:.2f} mm")
        
        info.append("=" * 50)
        return "\n".join(info)


class CameraCalibrationTool:
    def __init__(self, camera_index: int = 0, window_name: str = "Calibration Tool"):
        self.camera_index = camera_index
        self.window_name = window_name
        self.cap: Optional[cv2.VideoCapture] = None
        self.calibrator = Calibrator()
        
        self.click_points: List[Tuple[int, int]] = []
        self.roi_start: Optional[Tuple[int, int]] = None
        self.roi_end: Optional[Tuple[int, int]] = None
        self.is_selecting_roi = False
        self.current_frame: Optional[np.ndarray] = None
        self.mode = "click"
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

    def start_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"无法打开摄像头 (索引: {self.camera_index})")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"摄像头已打开 (索引: {self.camera_index})")
        return True

    def stop_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()
            print("摄像头已关闭")

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        
        for i, point in enumerate(self.click_points):
            cv2.circle(overlay, point, 5, (0, 255, 0), -1)
            cv2.circle(overlay, point, 8, (0, 255, 0), 1)
            
            label = f"P{i+1}"
            if self.calibrator.is_calibrated():
                try:
                    rx, ry = self.calibrator.pixel_to_robot(point[0], point[1])
                    label = f"P{i+1}:({rx:.0f},{ry:.0f})"
                except:
                    pass
            
            cv2.putText(overlay, label, (point[0] + 10, point[1] - 10),
                       self.font, self.font_scale, (0, 255, 0), self.font_thickness)
        
        for i, point in enumerate(self.calibrator.get_points()):
            pixel_point = (point.pixel_x, point.pixel_y)
            cv2.circle(overlay, pixel_point, 6, (255, 0, 0), -1)
            cv2.circle(overlay, pixel_point, 10, (255, 0, 0), 1)
            
            label = f"C{i+1}:{point.label}" if point.label else f"C{i+1}"
            info = f"{label} ({point.robot_x:.0f},{point.robot_y:.0f})"
            cv2.putText(overlay, info, (pixel_point[0] + 10, pixel_point[1] + 20),
                       self.font, self.font_scale, (255, 0, 0), self.font_thickness)
        
        if self.roi_start and self.roi_end:
            cv2.rectangle(overlay, self.roi_start, self.roi_end, (0, 0, 255), 2)
            
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(overlay, (center_x, center_y), 4, (0, 0, 255), -1)
            
            roi_label = f"ROI: ({x1},{y1})-({x2},{y2})"
            cv2.putText(overlay, roi_label, (x1, y1 - 10),
                       self.font, self.font_scale, (0, 0, 255), self.font_thickness)
            
            if self.calibrator.is_calibrated():
                try:
                    cx, cy = self.calibrator.pixel_to_robot(center_x, center_y)
                    center_label = f"Center: ({cx:.0f}, {cy:.0f})"
                    cv2.putText(overlay, center_label, (x1, y1 + 15),
                               self.font, self.font_scale, (0, 165, 255), self.font_thickness)
                except:
                    pass
        
        mode_text = "模式: 点击取点" if self.mode == "click" else "模式: 框选ROI"
        cv2.putText(overlay, mode_text, (10, 20),
                   self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        h, w = overlay.shape[:2]
        if self.calibrator.is_calibrated():
            try:
                tl_rx, tl_ry = self.calibrator.pixel_to_robot(0, 0)
                br_rx, br_ry = self.calibrator.pixel_to_robot(w, h)
                
                cv2.putText(overlay, f"({tl_rx:.0f},{tl_ry:.0f})", (5, 15),
                           self.font, self.font_scale, (0, 255, 255), self.font_thickness)
                cv2.putText(overlay, f"({br_rx:.0f},{br_ry:.0f})", (w - 100, h - 10),
                           self.font, self.font_scale, (0, 255, 255), self.font_thickness)
            except:
                pass
        
        help_text = "左键:取点/框选 | 'r':重置 | 'm':切换模式 | 's':保存 | 'q':退出"
        cv2.putText(overlay, help_text, (10, h - 10),
                   self.font, self.font_scale * 0.8, (200, 200, 200), self.font_thickness)
        
        return overlay

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        if self.mode == "click":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.click_points.append((x, y))
                print(f"点击取点: 像素坐标 ({x}, {y})")
                
                if self.calibrator.is_calibrated():
                    try:
                        rx, ry = self.calibrator.pixel_to_robot(x, y)
                        print(f"  转换为机械臂坐标: ({rx:.1f}, {ry:.1f})")
                    except Exception as e:
                        print(f"  坐标转换失败: {e}")
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_start = (x, y)
                self.is_selecting_roi = True
            elif event == cv2.EVENT_MOUSEMOVE and self.is_selecting_roi:
                self.roi_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.roi_end = (x, y)
                self.is_selecting_roi = False
                
                if self.roi_start and self.roi_end:
                    x1, y1 = self.roi_start
                    x2, y2 = self.roi_end
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    print(f"框选ROI: 左上角({x1}, {y1}), 右下角({x2}, {y2})")
                    print(f"  ROI中心点: ({center_x}, {center_y})")
                    
                    if self.calibrator.is_calibrated():
                        try:
                            cx, cy = self.calibrator.pixel_to_robot(center_x, center_y)
                            print(f"  中心点机械臂坐标: ({cx:.1f}, {cy:.1f})")
                        except Exception as e:
                            print(f"  坐标转换失败: {e}")

    def run_interactive(self) -> None:
        if not self.start_camera():
            return
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 50)
        print("视觉坐标标定与点击取点工具")
        print("=" * 50)
        print("操作说明:")
        print("  鼠标左键: 点击取点(点击模式) 或 框选ROI(ROI模式)")
        print("  按键 'r': 重置所有点击点和ROI")
        print("  按键 'm': 切换模式 (点击取点 / 框选ROI)")
        print("  按键 's': 保存当前帧和标定数据")
        print("  按键 'c': 执行标定(需要已添加的标定点)")
        print("  按键 'q': 退出")
        print("=" * 50 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取画面")
                break
            
            self.current_frame = frame.copy()
            display_frame = self._draw_overlay(frame)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("退出工具")
                break
            elif key == ord('r'):
                self.click_points.clear()
                self.roi_start = None
                self.roi_end = None
                print("已重置点击点和ROI")
            elif key == ord('m'):
                self.mode = "roi" if self.mode == "click" else "click"
                print(f"切换至{'框选ROI' if self.mode == 'roi' else '点击取点'}模式")
            elif key == ord('s'):
                timestamp = cv2.getTickCount()
                frame_filename = f"calibration_frame_{timestamp}.jpg"
                cv2.imwrite(frame_filename, self.current_frame)
                print(f"当前帧已保存至: {frame_filename}")
                
                if self.calibrator.get_points():
                    self.calibrator.save_calibration()
            elif key == ord('c'):
                if len(self.calibrator.get_points()) >= 2:
                    self.calibrator.calculate_linear_transform()
                    errors = self.calibrator.evaluate_errors()
                    print(f"\n标定完成!")
                    print(f"  平均误差: {errors['mean_error']:.2f} mm")
                    print(f"  最大误差: {errors['max_error']:.2f} mm")
                else:
                    print("需要至少2个标定点才能执行标定")
        
        self.stop_camera()

    def get_click_points(self) -> List[Tuple[int, int]]:
        return self.click_points.copy()

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        if self.roi_start and self.roi_end:
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return None


def create_default_calibrator() -> Calibrator:
    calibrator = Calibrator()
    
    calibrator.add_calibration_point((130, 290), (-21.8, -197.4), "左下角")
    calibrator.add_calibration_point((640, 0), (215.0, -59.1), "右上角")
    calibrator.calculate_linear_transform()
    
    return calibrator
