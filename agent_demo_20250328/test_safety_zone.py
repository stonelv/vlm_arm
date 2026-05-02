#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全禁区模块单元测试
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_safety_zone import (
    ZoneType,
    AlertLevel,
    SafetyZone,
    ObjectState,
    ZoneManager,
    PathPlanner,
    SafetyMonitor,
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

    def test_plan_alternate_path(self):
        start = (0, 0, 100)
        end = (200, 0, 100)
        
        waypoints = self.planner.plan_alternate_path(start, end)
        
        self.assertGreater(len(waypoints), 2)
        self.assertEqual(waypoints[0], start)
        self.assertEqual(waypoints[-1], end)

    def test_is_point_safe(self):
        self.assertTrue(self.planner.is_point_safe((0, 0, 100)))
        self.assertFalse(self.planner.is_point_safe((100, 0, 100)))


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

    def test_velocity_prediction(self):
        self.monitor.update_end_effector((90, 0, 0), velocity=(-200, 0, 0))
        
        self.monitor.start_monitoring()
        import time
        time.sleep(0.2)
        self.monitor.stop_monitoring()
        
        status = self.monitor.get_current_status()
        self.assertEqual(status["end_effector"]["position"], (90, 0, 0))


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
