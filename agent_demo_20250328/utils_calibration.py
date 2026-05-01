# utils_calibration.py
# 视觉坐标标定与点击取点工具
# 支持多标定点仿射变换、数据保存加载、误差评估
# 交互标定闭环：支持点击添加标定点、终端输入机械臂坐标

import cv2
import numpy as np
import json
import os
import sys
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
        self._use_linear_interpolation = False

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
        idx = len(self.calibration_points)
        label_str = f" [{point.label}]" if point.label else ""
        print(f"[标定点{idx}{label_str}] 像素({point.pixel_x}, {point.pixel_y}) -> 机械臂({point.robot_x:.1f}, {point.robot_y:.1f})")

    def remove_last_point(self) -> None:
        if self.calibration_points:
            removed = self.calibration_points.pop()
            print(f"已移除最后一个标定点: 像素({removed.pixel_x}, {removed.pixel_y})")
        else:
            print("没有标定点可移除")

    def clear_points(self) -> None:
        self.calibration_points.clear()
        self.transform_matrix = None
        self.affine_matrix = None
        self._use_linear_interpolation = False
        print("已清除所有标定点")

    def get_points(self) -> List[CalibrationPoint]:
        return self.calibration_points

    def calculate_linear_transform(self) -> bool:
        point_count = len(self.calibration_points)
        
        if point_count < 2:
            print("至少需要2个标定点进行标定")
            return False
        
        if point_count == 2:
            print(f"[2点标定] 使用线性插值模式（不调用 estimateAffine2D）")
            self._use_linear_interpolation = True
            self.transform_matrix = None
            self.affine_matrix = None
            print("2点标定完成（线性插值模式）")
            return True
        else:
            print(f"[{point_count}点标定] 使用仿射变换矩阵")
            self._use_linear_interpolation = False
            
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
                print("仿射变换矩阵计算失败")
                return False
            
            self.affine_matrix = self.transform_matrix.copy()
            print(f"{point_count}点标定完成（仿射变换模式）")
            return True

    def _pixel_to_robot_linear_interpolation(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        if len(self.calibration_points) != 2:
            raise ValueError("线性插值模式需要恰好2个标定点")
        
        p1, p2 = self.calibration_points
        
        X_cali_im = [p1.pixel_x, p2.pixel_x]
        X_cali_mc = [p1.robot_x, p2.robot_x]
        
        Y_cali_im = [p2.pixel_y, p1.pixel_y]
        Y_cali_mc = [p2.robot_y, p1.robot_y]
        
        X_mc = np.interp(pixel_x, X_cali_im, X_cali_mc)
        Y_mc = np.interp(pixel_y, Y_cali_im, Y_cali_mc)
        
        return float(X_mc), float(Y_mc)

    def _pixel_to_robot_affine(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        if self.transform_matrix is None:
            raise ValueError("仿射变换矩阵未计算")
        
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        robot_point = cv2.transform(pixel_point, self.transform_matrix)
        return float(robot_point[0][0][0]), float(robot_point[0][0][1])

    def pixel_to_robot(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        point_count = len(self.calibration_points)
        
        if point_count < 2:
            raise ValueError("标定未完成，至少需要2个标定点")
        
        if self._use_linear_interpolation or point_count == 2:
            return self._pixel_to_robot_linear_interpolation(pixel_x, pixel_y)
        else:
            return self._pixel_to_robot_affine(pixel_x, pixel_y)

    def robot_to_pixel(self, robot_x: float, robot_y: float) -> Tuple[int, int]:
        point_count = len(self.calibration_points)
        
        if point_count < 2:
            raise ValueError("标定未完成，至少需要2个标定点")
        
        if point_count == 2:
            p1, p2 = self.calibration_points
            
            X_cali_mc = [p1.robot_x, p2.robot_x]
            X_cali_im = [p1.pixel_x, p2.pixel_x]
            
            Y_cali_mc = [p2.robot_y, p1.robot_y]
            Y_cali_im = [p2.pixel_y, p1.pixel_y]
            
            px = int(np.interp(robot_x, X_cali_mc, X_cali_im))
            py = int(np.interp(robot_y, Y_cali_mc, Y_cali_im))
            return px, py
        else:
            if self.transform_matrix is None:
                raise ValueError("仿射变换矩阵未计算")
            
            inverse_matrix = cv2.invertAffineTransform(self.transform_matrix)
            robot_point = np.array([[[robot_x, robot_y]]], dtype=np.float32)
            pixel_point = cv2.transform(robot_point, inverse_matrix)
            return int(pixel_point[0][0][0]), int(pixel_point[0][0][1])

    def evaluate_errors(self) -> Dict[str, float]:
        point_count = len(self.calibration_points)
        
        if point_count < 2:
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
            "affine_matrix": self.affine_matrix.tolist() if self.affine_matrix is not None else None,
            "use_linear_interpolation": self._use_linear_interpolation
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
            
            self._use_linear_interpolation = data.get("use_linear_interpolation", False)
            
            point_count = len(self.calibration_points)
            mode = "线性插值" if (self._use_linear_interpolation or point_count == 2) else "仿射变换"
            print(f"已加载 {point_count} 个标定点，模式: {mode}")
            return True
        except Exception as e:
            print(f"加载标定数据失败: {e}")
            return False

    def is_calibrated(self) -> bool:
        point_count = len(self.calibration_points)
        if point_count == 2:
            return True
        return self.transform_matrix is not None

    def get_calibration_mode(self) -> str:
        point_count = len(self.calibration_points)
        if point_count < 2:
            return "未标定"
        elif point_count == 2 or self._use_linear_interpolation:
            return "线性插值(2点)"
        else:
            return f"仿射变换({point_count}点)"

    def get_calibration_info(self) -> str:
        info = ["=" * 60]
        info.append("标定信息")
        info.append("=" * 60)
        info.append(f"标定点数量: {len(self.calibration_points)}")
        info.append(f"标定模式: {self.get_calibration_mode()}")
        
        for i, p in enumerate(self.calibration_points, 1):
            label = f" [{p.label}]" if p.label else ""
            info.append(f"  [标定点{i}{label}] 像素({p.pixel_x}, {p.pixel_y}) -> 机械臂({p.robot_x:.1f}, {p.robot_y:.1f})")
        
        if self.is_calibrated():
            errors = self.evaluate_errors()
            if errors["mean_error"] >= 0:
                info.append(f"\n误差评估:")
                info.append(f"  平均误差: {errors['mean_error']:.2f} mm")
                info.append(f"  最大误差: {errors['max_error']:.2f} mm")
                info.append(f"  最小误差: {errors['min_error']:.2f} mm")
                info.append(f"  标准差:   {errors['std_error']:.2f} mm")
        
        info.append("=" * 60)
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
        
        self.status_message: str = ""
        self.status_color: Tuple[int, int, int] = (255, 255, 255)
        self.status_frame_count: int = 0

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

    def set_status(self, message: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        self.status_message = message
        self.status_color = color
        self.status_frame_count = 120

    def _input_robot_coords(self) -> Optional[Tuple[float, float]]:
        print("\n" + "=" * 50)
        print("添加标定点 - 输入机械臂坐标")
        print("=" * 50)
        print("格式: 直接输入 rx,ry 或 rx ry （用逗号或空格分隔）")
        print("示例: 100.5,-150.3 或 100.5 -150.3")
        print("输入 'q' 取消")
        print("-" * 50)
        
        try:
            user_input = input("请输入机械臂坐标 (rx, ry): ").strip()
            
            if user_input.lower() == 'q':
                print("已取消")
                return None
            
            user_input = user_input.replace(',', ' ')
            parts = user_input.split()
            
            if len(parts) != 2:
                print(f"错误: 需要2个数值，实际输入了 {len(parts)} 个")
                return None
            
            rx = float(parts[0])
            ry = float(parts[1])
            
            print(f"已输入: rx={rx:.2f}, ry={ry:.2f}")
            return (rx, ry)
            
        except ValueError as e:
            print(f"输入格式错误: {e}")
            return None
        except KeyboardInterrupt:
            print("\n已取消")
            return None

    def _add_click_point_as_calibration(self) -> bool:
        if not self.click_points:
            self.set_status("错误: 没有点击的点，请先点击图像", (0, 0, 255))
            print("错误: 没有点击的点，请先用鼠标点击图像取点")
            return False
        
        last_point = self.click_points[-1]
        px, py = last_point
        
        print(f"\n准备添加标定点: 像素坐标 ({px}, {py})")
        
        robot_coords = self._input_robot_coords()
        if robot_coords is None:
            self.set_status("已取消添加标定点", (0, 165, 255))
            return False
        
        rx, ry = robot_coords
        
        point_idx = len(self.calibrator.get_points()) + 1
        label = f"P{point_idx}"
        
        self.calibrator.add_calibration_point(
            pixel_coord=(px, py),
            robot_coord=(rx, ry),
            label=label
        )
        
        point_count = len(self.calibrator.get_points())
        if point_count >= 2:
            self.calibrator.calculate_linear_transform()
            mode = self.calibrator.get_calibration_mode()
            errors = self.calibrator.evaluate_errors()
            self.set_status(
                f"已添加标定点{point_idx} - {mode} - 平均误差: {errors['mean_error']:.1f}mm",
                (0, 255, 0)
            )
        else:
            self.set_status(f"已添加标定点{point_idx}，还需至少{2 - point_count}个点", (0, 255, 255))
        
        return True

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        for i, point in enumerate(self.click_points):
            cv2.circle(overlay, point, 5, (0, 255, 0), -1)
            cv2.circle(overlay, point, 8, (0, 255, 0), 1)
            
            label = f"Click-{i+1}"
            if self.calibrator.is_calibrated():
                try:
                    rx, ry = self.calibrator.pixel_to_robot(point[0], point[1])
                    label = f"Click-{i+1}:({rx:.0f},{ry:.0f})"
                except:
                    pass
            
            cv2.putText(overlay, label, (point[0] + 10, point[1] - 10),
                       self.font, self.font_scale, (0, 255, 0), self.font_thickness)
        
        for i, point in enumerate(self.calibrator.get_points(), 1):
            pixel_point = (point.pixel_x, point.pixel_y)
            
            cv2.circle(overlay, pixel_point, 10, (255, 0, 0), 2)
            cv2.circle(overlay, pixel_point, 6, (255, 0, 0), -1)
            cv2.line(overlay, 
                     (pixel_point[0] - 12, pixel_point[1]), 
                     (pixel_point[0] + 12, pixel_point[1]), 
                     (0, 0, 255), 2)
            cv2.line(overlay, 
                     (pixel_point[0], pixel_point[1] - 12), 
                     (pixel_point[0], pixel_point[1] + 12), 
                     (0, 0, 255), 2)
            
            label = point.label if point.label else f"C{i}"
            info_top = f"{label}"
            info_bottom = f"({point.robot_x:.0f},{point.robot_y:.0f})"
            
            cv2.putText(overlay, info_top, (pixel_point[0] + 15, pixel_point[1] - 5),
                       self.font, self.font_scale, (255, 0, 0), self.font_thickness)
            cv2.putText(overlay, info_bottom, (pixel_point[0] + 15, pixel_point[1] + 15),
                       self.font, self.font_scale, (0, 165, 255), self.font_thickness)
        
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
        
        calib_status = f"标定: {self.calibrator.get_calibration_mode()}"
        calib_color = (0, 255, 0) if self.calibrator.is_calibrated() else (0, 165, 255)
        cv2.putText(overlay, calib_status, (10, 40),
                   self.font, self.font_scale, calib_color, self.font_thickness)
        
        point_count = len(self.calibrator.get_points())
        click_count = len(self.click_points)
        status_text = f"标定点: {point_count} | 点击点: {click_count}"
        cv2.putText(overlay, status_text, (10, 60),
                   self.font, self.font_scale, (200, 200, 200), self.font_thickness)
        
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
        
        if self.status_message and self.status_frame_count > 0:
            y_pos = h - 50
            text_size = cv2.getTextSize(self.status_message, self.font, self.font_scale, self.font_thickness)[0]
            x_pos = (w - text_size[0]) // 2
            
            cv2.rectangle(overlay, 
                         (x_pos - 10, y_pos - text_size[1] - 10),
                         (x_pos + text_size[0] + 10, y_pos + 10),
                         (0, 0, 0), -1)
            cv2.putText(overlay, self.status_message, (x_pos, y_pos),
                       self.font, self.font_scale, self.status_color, self.font_thickness)
            
            self.status_frame_count -= 1
        
        help_lines = [
            "鼠标:取点/框选 | 'a':添加标定点 | 'd':删标定点 | 'r':重置",
            "'m':切换模式 | 'c':计算标定 | 's':保存 | 'l':加载 | 'q':退出"
        ]
        for i, line in enumerate(help_lines):
            cv2.putText(overlay, line, (10, h - 10 - i * 18),
                       self.font, self.font_scale * 0.8, (150, 150, 150), self.font_thickness)
        
        return overlay

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        if self.mode == "click":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.click_points.append((x, y))
                click_idx = len(self.click_points)
                print(f"[点击-{click_idx}] 像素坐标: ({x}, {y})")
                
                if self.calibrator.is_calibrated():
                    try:
                        rx, ry = self.calibrator.pixel_to_robot(x, y)
                        print(f"         -> 机械臂坐标: ({rx:.1f}, {ry:.1f})")
                    except Exception as e:
                        print(f"         -> 坐标转换失败: {e}")
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
                    
                    print(f"[ROI框选] 左上角({x1}, {y1}) -> 右下角({x2}, {y2})")
                    print(f"         中心点像素坐标: ({center_x}, {center_y})")
                    
                    if self.calibrator.is_calibrated():
                        try:
                            cx, cy = self.calibrator.pixel_to_robot(center_x, center_y)
                            print(f"         中心点机械臂坐标: ({cx:.1f}, {cy:.1f})")
                        except Exception as e:
                            print(f"         坐标转换失败: {e}")

    def run_interactive(self) -> None:
        if not self.start_camera():
            return
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("视觉坐标标定与点击取点工具 - 交互标定闭环")
        print("=" * 60)
        print("操作说明:")
        print("  鼠标左键: 点击取点(点击模式) 或 框选ROI(ROI模式)")
        print("  按键 'a': 添加最后一个点击点为标定点（需输入机械臂坐标）")
        print("  按键 'd': 删除最后一个标定点")
        print("  按键 'r': 重置所有点击点和ROI")
        print("  按键 'm': 切换模式 (点击取点 / 框选ROI)")
        print("  按键 'c': 执行标定计算")
        print("  按键 's': 保存当前帧和标定数据")
        print("  按键 'l': 从默认文件加载标定数据")
        print("  按键 'q': 退出")
        print("=" * 60)
        print("标定流程提示:")
        print("  1. 鼠标点击图像上的标记点取像素坐标")
        print("  2. 按 'a' 键，输入该点对应的机械臂坐标")
        print("  3. 重复步骤1-2，至少添加2个标定点")
        print("  4. 添加第2个点后自动执行标定（2点线性插值模式）")
        print("  5. ≥3点时使用仿射变换，按 'c' 可重新计算")
        print("  6. 标定完成后，新点击的点会自动显示机械臂坐标")
        print("=" * 60 + "\n")
        
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
            
            elif key == ord('a'):
                print("\n>>> 按下 'a' - 添加标定点")
                self._add_click_point_as_calibration()
            
            elif key == ord('d'):
                print("\n>>> 按下 'd' - 删除最后一个标定点")
                point_count_before = len(self.calibrator.get_points())
                self.calibrator.remove_last_point()
                point_count_after = len(self.calibrator.get_points())
                
                if point_count_after != point_count_before:
                    if point_count_after >= 2:
                        self.calibrator.calculate_linear_transform()
                        self.set_status(f"已删除标定点，剩余 {point_count_after} 个", (0, 255, 255))
                    else:
                        self.set_status(f"已删除标定点，剩余 {point_count_after} 个（需≥2个标定）", (0, 165, 255))
            
            elif key == ord('r'):
                self.click_points.clear()
                self.roi_start = None
                self.roi_end = None
                print("已重置点击点和ROI")
                self.set_status("已重置点击点和ROI", (255, 255, 255))
            
            elif key == ord('m'):
                self.mode = "roi" if self.mode == "click" else "click"
                mode_name = "框选ROI" if self.mode == "roi" else "点击取点"
                print(f"切换至{mode_name}模式")
                self.set_status(f"切换至{mode_name}模式", (255, 255, 255))
            
            elif key == ord('c'):
                print("\n>>> 按下 'c' - 执行标定计算")
                point_count = len(self.calibrator.get_points())
                if point_count >= 2:
                    success = self.calibrator.calculate_linear_transform()
                    if success:
                        mode = self.calibrator.get_calibration_mode()
                        errors = self.calibrator.evaluate_errors()
                        print(f"标定完成！模式: {mode}")
                        print(f"平均误差: {errors['mean_error']:.2f} mm")
                        print(f"最大误差: {errors['max_error']:.2f} mm")
                        self.set_status(
                            f"{mode} - 平均误差: {errors['mean_error']:.1f}mm",
                            (0, 255, 0)
                        )
                else:
                    print(f"需要至少2个标定点，当前只有 {point_count} 个")
                    self.set_status(f"需要至少2个标定点，当前 {point_count} 个", (0, 0, 255))
            
            elif key == ord('s'):
                timestamp = cv2.getTickCount()
                frame_filename = f"calibration_frame_{timestamp}.jpg"
                if self.current_frame is not None:
                    cv2.imwrite(frame_filename, self.current_frame)
                    print(f"当前帧已保存至: {frame_filename}")
                
                if self.calibrator.get_points():
                    self.calibrator.save_calibration()
                    self.set_status("已保存标定数据", (0, 255, 0))
            
            elif key == ord('l'):
                print("\n>>> 按下 'l' - 加载标定数据")
                success = self.calibrator.load_calibration()
                if success:
                    self.set_status(
                        f"已加载 {len(self.calibrator.get_points())} 个标定点 - {self.calibrator.get_calibration_mode()}",
                        (0, 255, 0)
                    )
                else:
                    self.set_status("加载标定数据失败", (0, 0, 255))
        
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
