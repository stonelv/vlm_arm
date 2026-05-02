#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全禁区模块单元测试 - 完整版
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_safety_zone import (
    ZoneType,
    AlertLevel,
    SafetyZone,
    ObjectState,
    ZoneManager,
    PathPlanner,
    SafetyMonitor,
    SafeArmController,
    VLMConfig,
    VLMDetector,
    create_default_zone_manager,
    alert_callback_example
)


class TestSafetyZone(unittest.TestCase):
    
    def test_sphere_zone_is_inside(self):
        sphere = SafetyZone(
            name="测试球形禁区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0),
            warning_distance=20,
            danger_distance=10
        )
        
        self.assertTrue(sphere.is_inside((0, 0, 0)))
        self.assertTrue(sphere.is_inside((30, 0, 0)))
        self.assertTrue(sphere.is_inside((0, 40, 0)))
        self.assertFalse(sphere.is_inside((60, 0, 0)))
        self.assertFalse(sphere.is_inside((100, 100, 100)))

    def test_rectangle_zone_is_inside(self):
        rectangle = SafetyZone(
            name="测试矩形禁区",
            zone_type=ZoneType.RECTANGLE,
            center=(100, 100, 100),
            dimensions=(100, 100, 100),
            warning_distance=20,
            danger_distance=10
        )
        
        self.assertTrue(rectangle.is_inside((100, 100, 100)))
        self.assertTrue(rectangle.is_inside((120, 80, 110)))
        self.assertFalse(rectangle.is_inside((200, 100, 100)))
        self.assertFalse(rectangle.is_inside((100, 100, 200)))

    def test_cylinder_zone_is_inside(self):
        cylinder = SafetyZone(
            name="测试圆柱形禁区",
            zone_type=ZoneType.CYLINDER,
            center=(50, 50, 100),
            dimensions=(30, 100, 0),
            warning_distance=20,
            danger_distance=10
        )
        
        self.assertTrue(cylinder.is_inside((50, 50, 100)))
        self.assertTrue(cylinder.is_inside((60, 50, 120)))
        self.assertFalse(cylinder.is_inside((100, 50, 100)))
        self.assertFalse(cylinder.is_inside((50, 50, 200)))

    def test_distance_to_zone(self):
        sphere = SafetyZone(
            name="测试球形禁区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0),
            warning_distance=20,
            danger_distance=10
        )
        
        self.assertEqual(sphere.distance_to_zone((0, 0, 0)), 0.0)
        self.assertEqual(sphere.distance_to_zone((100, 0, 0)), 50.0)
        self.assertAlmostEqual(sphere.distance_to_zone((30, 40, 0)), 0.0)
        self.assertAlmostEqual(sphere.distance_to_zone((60, 80, 0)), 50.0)

    def test_get_alert_level(self):
        sphere = SafetyZone(
            name="测试球形禁区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0),
            warning_distance=30,
            danger_distance=15
        )
        
        self.assertEqual(sphere.get_alert_level((0, 0, 0)), AlertLevel.EMERGENCY)
        self.assertEqual(sphere.get_alert_level((55, 0, 0)), AlertLevel.DANGER)
        self.assertEqual(sphere.get_alert_level((70, 0, 0)), AlertLevel.WARNING)
        self.assertEqual(sphere.get_alert_level((100, 0, 0)), AlertLevel.SAFE)

    def test_zone_disabled(self):
        sphere = SafetyZone(
            name="测试禁用禁区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0),
            enabled=False
        )
        
        self.assertFalse(sphere.is_inside((0, 0, 0)))
        self.assertEqual(sphere.distance_to_zone((0, 0, 0)), float('inf'))
        self.assertEqual(sphere.get_alert_level((0, 0, 0)), AlertLevel.SAFE)


class TestZoneManager(unittest.TestCase):
    
    def setUp(self):
        self.manager = ZoneManager()
        
    def test_add_and_get_zone(self):
        zone = SafetyZone(
            name="测试区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0)
        )
        
        self.manager.add_zone(zone)
        retrieved = self.manager.get_zone("测试区")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "测试区")

    def test_remove_zone(self):
        zone = SafetyZone(
            name="待删除区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0)
        )
        
        self.manager.add_zone(zone)
        self.assertTrue(self.manager.remove_zone("待删除区"))
        self.assertFalse(self.manager.remove_zone("不存在的区"))
        self.assertIsNone(self.manager.get_zone("待删除区"))

    def test_update_zone(self):
        zone = SafetyZone(
            name="待更新区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0),
            enabled=True
        )
        
        self.manager.add_zone(zone)
        self.assertTrue(self.manager.update_zone("待更新区", enabled=False))
        
        retrieved = self.manager.get_zone("待更新区")
        self.assertFalse(retrieved.enabled)

    def test_check_all_zones(self):
        zone1 = SafetyZone(
            name="近区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0),
            warning_distance=30,
            danger_distance=15
        )
        
        zone2 = SafetyZone(
            name="远区",
            zone_type=ZoneType.SPHERE,
            center=(200, 0, 0),
            dimensions=(50, 0, 0)
        )
        
        self.manager.add_zone(zone1)
        self.manager.add_zone(zone2)
        
        highest, alerts = self.manager.check_all_zones((30, 0, 0))
        self.assertEqual(highest, AlertLevel.EMERGENCY)
        
        highest, alerts = self.manager.check_all_zones((100, 0, 0))
        self.assertEqual(highest, AlertLevel.SAFE)

    def test_get_closest_zone(self):
        zone1 = SafetyZone(
            name="近区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0)
        )
        
        zone2 = SafetyZone(
            name="远区",
            zone_type=ZoneType.SPHERE,
            center=(200, 0, 0),
            dimensions=(50, 0, 0)
        )
        
        self.manager.add_zone(zone1)
        self.manager.add_zone(zone2)
        
        closest, distance = self.manager.get_closest_zone((25, 0, 0))
        self.assertEqual(closest.name, "近区")
        self.assertEqual(distance, 0.0)
        
        closest, distance = self.manager.get_closest_zone((150, 0, 0))
        self.assertEqual(closest.name, "远区")
        self.assertEqual(distance, 0.0)

    def test_create_default_zone_manager(self):
        manager = create_default_zone_manager()
        zones = manager.list_zones()
        
        self.assertEqual(len(zones), 2)
        zone_names = [z.name for z in zones]
        self.assertIn("桌面角落禁区", zone_names)
        self.assertIn("中心支柱禁区", zone_names)


class TestPathPlanner(unittest.TestCase):
    
    def setUp(self):
        self.zone_manager = ZoneManager()
        self.obstacle = SafetyZone(
            name="障碍物",
            zone_type=ZoneType.SPHERE,
            center=(100, 0, 100),
            dimensions=(30, 0, 0),
            warning_distance=20,
            danger_distance=10
        )
        self.zone_manager.add_zone(self.obstacle)
        self.planner = PathPlanner(self.zone_manager)

    def test_check_path_safe_clear(self):
        is_safe, alerts = self.planner.check_path_safe((0, 0, 100), (200, 200, 100))
        self.assertTrue(is_safe)

    def test_check_path_safe_blocked(self):
        is_safe, alerts = self.planner.check_path_safe((0, 0, 100), (200, 0, 100))
        self.assertFalse(is_safe)

    def test_plan_alternate_path_basic(self):
        start = (0, 0, 100)
        end = (200, 0, 100)
        
        waypoints = self.planner.plan_alternate_path(start, end)
        
        self.assertEqual(waypoints[0], start)
        self.assertEqual(waypoints[-1], end)
        self.assertGreater(len(waypoints), 2)

    def test_arc_waypoints_generation(self):
        start = (0, 0, 100)
        end = (200, 0, 100)
        obstacle_center = (100, 0, 100)
        obstacle_radius = 30
        
        clockwise_waypoints = self.planner._generate_arc_waypoints(
            start, end, obstacle_center, obstacle_radius, clockwise=True
        )
        
        counter_clockwise_waypoints = self.planner._generate_arc_waypoints(
            start, end, obstacle_center, obstacle_radius, clockwise=False
        )
        
        self.assertEqual(clockwise_waypoints[0], start)
        self.assertEqual(clockwise_waypoints[-1], end)
        
        self.assertGreater(len(clockwise_waypoints), 2)
        
        safe_radius = obstacle_radius + self.planner.min_detour_distance
        for wp in clockwise_waypoints:
            dx = wp[0] - obstacle_center[0]
            dy = wp[1] - obstacle_center[1]
            dist = np.sqrt(dx**2 + dy**2)
            self.assertGreater(dist, obstacle_radius - 1e-6)

    def test_get_obstacle_info(self):
        start = (0, 0, 100)
        end = (200, 0, 100)
        
        obstacles = self.planner._get_obstacle_info(start, end)
        
        self.assertEqual(len(obstacles), 1)
        self.assertEqual(obstacles[0][0].name, "障碍物")

    def test_is_point_safe(self):
        self.assertTrue(self.planner.is_point_safe((0, 0, 100)))
        self.assertFalse(self.planner.is_point_safe((100, 0, 100)))

    def test_plan_smooth_path(self):
        start = (0, 0, 100)
        end = (200, 0, 100)
        
        smooth_waypoints = self.planner.plan_smooth_path(start, end, num_waypoints=5)
        
        self.assertEqual(smooth_waypoints[0], start)
        self.assertEqual(smooth_waypoints[-1], end)

    def test_estimate_path_length(self):
        waypoints = [(0, 0, 0), (100, 0, 0), (100, 100, 0)]
        
        length = self.planner.estimate_path_length(waypoints)
        
        self.assertAlmostEqual(length, 200.0)


class TestSafetyMonitor(unittest.TestCase):
    
    def setUp(self):
        self.zone_manager = ZoneManager()
        self.zone_manager.add_zone(SafetyZone(
            name="测试禁区",
            zone_type=ZoneType.SPHERE,
            center=(0, 0, 0),
            dimensions=(50, 0, 0),
            warning_distance=30,
            danger_distance=15
        ))
        self.path_planner = PathPlanner(self.zone_manager)
        self.monitor = SafetyMonitor(self.zone_manager, self.path_planner)
        self.callback_calls = []
        
    def test_update_end_effector(self):
        self.monitor.update_end_effector((100, 100, 100))
        status = self.monitor.get_current_status()
        
        self.assertEqual(status["end_effector"]["position"], (100, 100, 100))
        self.assertEqual(status["end_effector"]["alert_level"], "safe")

    def test_update_target(self):
        self.monitor.update_target((0, 0, 0))
        status = self.monitor.get_current_status()
        
        self.assertEqual(status["target"]["position"], (0, 0, 0))
        self.assertEqual(status["target"]["alert_level"], "emergency")

    def test_alert_callback(self):
        def my_callback(level, zone_name, obj):
            self.callback_calls.append((level, zone_name, obj.name))
        
        self.monitor.add_alert_callback(my_callback)
        self.monitor.update_end_effector((0, 0, 0))
        
        self.assertEqual(len(self.callback_calls), 1)
        self.assertEqual(self.callback_calls[0][0], AlertLevel.EMERGENCY)
        self.assertEqual(self.callback_calls[0][1], "测试禁区")
        self.assertEqual(self.callback_calls[0][2], "end_effector")

    def test_monitor_start_stop(self):
        self.monitor.start_monitoring()
        time.sleep(0.1)
        self.monitor.stop_monitoring()
        
        status = self.monitor.get_current_status()
        self.assertIsNotNone(status)


class TestSafeArmController(unittest.TestCase):
    
    def setUp(self):
        self.mock_mc = MagicMock()
        self.mock_mc.get_coords.return_value = [0, 0, 220, 0, 180, 90]
        self.mock_mc.send_coords.return_value = 1
        self.mock_mc.get_angles.return_value = [0, 0, 0, 0, 0, 0]
        self.mock_mc.send_angles.return_value = 1
        self.mock_mc.send_angle.return_value = 1
        self.mock_mc.send_coord.return_value = 1
        self.mock_mc.release_all_servos.return_value = 1
        self.mock_mc.power_on.return_value = 1
        self.mock_mc.power_off.return_value = 1
        self.mock_mc.is_power_on.return_value = 1
        self.mock_mc.get_fresh_mode.return_value = 0
        self.mock_mc.set_fresh_mode.return_value = 1
        self.mock_mc.get_robot_status.return_value = [0, 0, 0, 0, 0, 0]
        self.mock_mc.read_next_error.return_value = [0, 0, 0, 0, 0, 0]
        self.mock_mc.is_controller_connected.return_value = 1
        self.mock_mc.get_system_version.return_value = 3.0
        self.mock_mc.get_atom_version.return_value = 2.0
        self.mock_mc.focus_servo.return_value = 1
        self.mock_mc.focus_all_servos.return_value = 1
        
        self.zone_manager = ZoneManager()
        self.path_planner = PathPlanner(self.zone_manager)
        self.monitor = SafetyMonitor(self.zone_manager, self.path_planner)
        self.controller = SafeArmController(self.mock_mc, self.zone_manager, self.monitor)

    def test_send_coords_safe_path(self):
        coords = [200, -100, 150, 0, 180, 90]
        speed = 20
        mode = 0
        
        result = self.controller.send_coords(coords, speed, mode)
        
        self.assertEqual(result, 1)
        self.mock_mc.send_coords.assert_called_once()

    def test_send_coords_blocked_target(self):
        blocked_zone = SafetyZone(
            name="目标区",
            zone_type=ZoneType.SPHERE,
            center=(200, -100, 150),
            dimensions=(50, 0, 0)
        )
        self.zone_manager.add_zone(blocked_zone)
        
        coords = [200, -100, 150, 0, 180, 90]
        result = self.controller.send_coords(coords, 20, 0)
        
        self.assertEqual(result, 0)

    def test_get_coords(self):
        coords = self.controller.get_coords()
        
        self.mock_mc.get_coords.assert_called_once()
        self.assertEqual(coords, [0, 0, 220, 0, 180, 90])

    def test_send_angles(self):
        angles = [0, 0, 0, 0, 0, 0]
        speed = 50
        
        result = self.controller.send_angles(angles, speed)
        
        self.assertEqual(result, 1)
        self.mock_mc.send_angles.assert_called_once_with(angles, speed)

    def test_send_angle(self):
        joint_id = 1
        angle = 45
        speed = 50
        
        result = self.controller.send_angle(joint_id, angle, speed)
        
        self.assertEqual(result, 1)
        self.mock_mc.send_angle.assert_called_once_with(joint_id, angle, speed)

    def test_get_angles(self):
        angles = self.controller.get_angles()
        
        self.mock_mc.get_angles.assert_called_once()
        self.assertEqual(angles, [0, 0, 0, 0, 0, 0])

    def test_send_coord(self):
        coord_id = 1
        coord_value = 150
        speed = 50
        
        result = self.controller.send_coord(coord_id, coord_value, speed)
        
        self.assertEqual(result, 1)
        self.mock_mc.send_coord.assert_called_once_with(coord_id, coord_value, speed)

    def test_emergency_stop(self):
        self.controller.emergency_stop()
        
        self.assertTrue(self.controller.emergency_stop_triggered)
        self.mock_mc.release_all_servos.assert_called_once()

    def test_emergency_stop_blocks_movement(self):
        self.controller.emergency_stop()
        
        coords = [200, -100, 150, 0, 180, 90]
        result = self.controller.send_coords(coords, 20, 0)
        
        self.assertEqual(result, 0)

    def test_reset_emergency(self):
        self.controller.emergency_stop()
        self.assertTrue(self.controller.emergency_stop_triggered)
        
        self.controller.reset_emergency()
        self.assertFalse(self.controller.emergency_stop_triggered)

    def test_power_management(self):
        result_on = self.controller.power_on()
        self.assertEqual(result_on, 1)
        
        result_off = self.controller.power_off()
        self.assertEqual(result_off, 1)
        
        result_is_on = self.controller.is_power_on()
        self.assertEqual(result_is_on, 1)

    def test_fresh_mode(self):
        mode = self.controller.get_fresh_mode()
        self.assertEqual(mode, 0)
        
        result = self.controller.set_fresh_mode(1)
        self.assertEqual(result, 1)

    def test_system_info(self):
        version = self.controller.get_system_version()
        self.assertEqual(version, 3.0)
        
        atom_version = self.controller.get_atom_version()
        self.assertEqual(atom_version, 2.0)
        
        status = self.controller.get_robot_status()
        self.assertEqual(status, [0, 0, 0, 0, 0, 0])
        
        error = self.controller.read_next_error()
        self.assertEqual(error, [0, 0, 0, 0, 0, 0])
        
        connected = self.controller.is_controller_connected()
        self.assertEqual(connected, 1)

    def test_focus_servos(self):
        result_single = self.controller.focus_servo(1)
        self.assertEqual(result_single, 1)
        
        result_all = self.controller.focus_all_servos()
        self.assertEqual(result_all, 1)

    def test_release_all_servos(self):
        result = self.controller.release_all_servos()
        self.assertEqual(result, 1)
        
        result_with_data = self.controller.release_all_servos(1)
        self.assertEqual(result_with_data, 1)


class TestVLMConfig(unittest.TestCase):
    
    def test_vlm_config_defaults(self):
        config = VLMConfig()
        
        self.assertEqual(config.api_key, "")
        self.assertEqual(config.base_url, "")
        self.assertEqual(config.model, "gpt-4o")
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.temperature, 0.0)

    def test_vlm_config_custom(self):
        config = VLMConfig(
            api_key="test-key",
            base_url="https://test.api.com",
            model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.7
        )
        
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://test.api.com")
        self.assertEqual(config.model, "gpt-4o-mini")
        self.assertEqual(config.max_tokens, 500)
        self.assertEqual(config.temperature, 0.7)


class TestIntegration(unittest.TestCase):
    
    def test_full_workflow(self):
        zone_manager = create_default_zone_manager()
        path_planner = PathPlanner(zone_manager)
        monitor = SafetyMonitor(zone_manager, path_planner)
        
        monitor.add_alert_callback(alert_callback_example)
        
        test_zone = SafetyZone(
            name="动态测试区",
            zone_type=ZoneType.SPHERE,
            center=(150, -100, 100),
            dimensions=(40, 0, 0),
            warning_distance=25,
            danger_distance=10
        )
        zone_manager.add_zone(test_zone)
        
        monitor.update_end_effector((200, -50, 220))
        status = monitor.get_current_status()
        self.assertEqual(status["end_effector"]["alert_level"], "safe")
        
        monitor.update_end_effector((150, -100, 100))
        status = monitor.get_current_status()
        self.assertEqual(status["end_effector"]["alert_level"], "emergency")
        
        start = (0, 0, 220)
        end = (200, -100, 100)
        
        is_safe, alerts = path_planner.check_path_safe(start, end)
        
        if not is_safe:
            waypoints = path_planner.plan_alternate_path(start, end)
            self.assertGreater(len(waypoints), 2)
        
        zones = zone_manager.list_zones()
        self.assertEqual(len(zones), 3)


class TestAlertCallbackExample(unittest.TestCase):
    
    def test_alert_callback_example_runs(self):
        obj = ObjectState(
            name="test_object",
            position=(0, 0, 0)
        )
        
        try:
            alert_callback_example(AlertLevel.WARNING, "测试区", obj)
            alert_callback_example(AlertLevel.DANGER, "测试区", obj)
            alert_callback_example(AlertLevel.EMERGENCY, "测试区", obj)
            alert_callback_example(AlertLevel.SAFE, "测试区", obj)
        except Exception as e:
            self.fail(f"alert_callback_example 抛出异常: {e}")


import time

if __name__ == '__main__':
    unittest.main(verbosity=2)
