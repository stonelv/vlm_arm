#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境自检与配置向导脚本
检查 Python 版本、requirements 依赖、摄像头/麦克风/扬声器可用性、机械臂连接状态
生成诊断报告（JSON + 终端摘要）
支持 --fix 自动写入/更新配置模板
"""

import sys
import os
import json
import re
import argparse
import platform
import shutil
from pathlib import Path
from datetime import datetime


MIN_PYTHON_VERSION = (3, 12)
REQUIRED_KEYS = ['Qwen_KEY', 'YI_KEY', 'QIANFAN_ACCESS_KEY', 
                  'QIANFAN_SECRET_KEY', 'APPBUILDER_TOKEN']
PLACEHOLDER_PATTERNS = [
    r'your_.*_here',
    r'XX+',
    r'xx+',
    r'\*{3,}',
]


class EnvironmentChecker:
    def __init__(self, project_root=None, config_path=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.agent_dir = self.project_root / 'agent_demo_20250328'
        
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.agent_dir / 'API_KEY.py'
        
        self.template_path = self.project_root / 'API_KEY.example.py'
        
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'python': {},
            'dependencies': {},
            'hardware': {},
            'robot_arm': {},
            'config': {},
            'summary': {
                'total_checks': 0,
                'passed': 0,
                'warnings': 0,
                'failed': 0
            }
        }
        self._issues = []

    def _check(self, name, passed, message=None, warning=False):
        self.report['summary']['total_checks'] += 1
        if passed:
            self.report['summary']['passed'] += 1
        elif warning:
            self.report['summary']['warnings'] += 1
        else:
            self.report['summary']['failed'] += 1
            if message:
                self._issues.append(f"[ERROR] {name}: {message}")
        return passed

    def check_python_version(self):
        print("\n" + "="*60)
        print("检查 Python 版本...")
        print("="*60)
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        self.report['python']['version'] = version_str
        self.report['python']['executable'] = sys.executable
        self.report['python']['platform'] = platform.system()
        
        is_valid = version >= MIN_PYTHON_VERSION
        min_ver_str = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
        
        self._check(
            'python_version',
            is_valid,
            f"需要 Python >= {min_ver_str}，当前版本 {version_str}"
        )
        
        if is_valid:
            print(f"✓ Python 版本: {version_str}")
        else:
            print(f"✗ Python 版本: {version_str} (需要 >= {min_ver_str})")
        
        self.report['python']['valid'] = is_valid
        return is_valid

    def _get_package_distribution(self, pkg_name):
        try:
            from importlib.metadata import distribution, PackageNotFoundError
            try:
                dist = distribution(pkg_name)
                return {
                    'status': 'installed',
                    'version': dist.version,
                    'name': dist.metadata.get('Name', pkg_name)
                }
            except PackageNotFoundError:
                pass
        except ImportError:
            try:
                import pkg_resources
                try:
                    dist = pkg_resources.get_distribution(pkg_name)
                    return {
                        'status': 'installed',
                        'version': dist.version,
                        'name': dist.project_name
                    }
                except pkg_resources.DistributionNotFound:
                    pass
            except ImportError:
                pass
        
        return {'status': 'missing', 'version': None}

    def check_dependencies(self):
        print("\n" + "="*60)
        print("检查 Python 依赖包...")
        print("="*60)
        
        requirements_path = self.project_root / 'requirements.txt'
        self.report['dependencies']['requirements_file'] = str(requirements_path)
        self.report['dependencies']['packages'] = {}
        
        if not requirements_path.exists():
            self._check('requirements_file', False, f"找不到 {requirements_path}")
            print(f"✗ 找不到 requirements.txt: {requirements_path}")
            return False
        
        package_names = []
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                pkg_name = line.split('==')[0].split('#')[0].strip()
                if pkg_name:
                    package_names.append(pkg_name)
        
        installed_packages = []
        missing_packages = []
        
        for pkg_name in package_names:
            result = self._get_package_distribution(pkg_name)
            
            self.report['dependencies']['packages'][pkg_name] = result
            
            if result['status'] == 'installed':
                installed_packages.append(pkg_name)
                print(f"✓ {pkg_name}: {result['version']}")
            else:
                missing_packages.append(pkg_name)
                print(f"✗ {pkg_name}: 未安装")
        
        self.report['dependencies']['installed_count'] = len(installed_packages)
        self.report['dependencies']['missing_count'] = len(missing_packages)
        self.report['dependencies']['missing_list'] = missing_packages
        
        if missing_packages:
            self._check('dependencies', False, f"缺少 {len(missing_packages)} 个依赖包")
            print(f"\n⚠  缺失的依赖包列表:")
            for pkg in missing_packages:
                print(f"   - {pkg}")
            print(f"\n安装命令: pip install {' '.join(missing_packages)}")
        else:
            self._check('dependencies', True)
            print(f"\n✓ 所有 {len(installed_packages)} 个依赖包已安装")
        
        self.report['dependencies']['valid'] = len(missing_packages) == 0
        return len(missing_packages) == 0

    def check_camera(self):
        print("\n" + "="*60)
        print("检查摄像头可用性...")
        print("="*60)
        
        self.report['hardware']['camera'] = {}
        
        try:
            import cv2
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                self.report['hardware']['camera']['status'] = 'unavailable'
                self._check('camera', False, "无法打开摄像头")
                print("✗ 无法打开摄像头")
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                self.report['hardware']['camera']['status'] = 'available'
                self.report['hardware']['camera']['resolution'] = f"{frame.shape[1]}x{frame.shape[0]}"
                print(f"✓ 摄像头可用，分辨率: {frame.shape[1]}x{frame.shape[0]}")
                self._check('camera', True)
                return True
            else:
                self.report['hardware']['camera']['status'] = 'no_frame'
                self._check('camera', False, "摄像头无法捕获画面")
                print("✗ 摄像头无法捕获画面")
                return False
                
        except ImportError:
            self.report['hardware']['camera']['status'] = 'opencv_missing'
            self._check('camera', False, "OpenCV 未安装，无法检查摄像头")
            print("✗ OpenCV 未安装，无法检查摄像头")
            return False
        except Exception as e:
            self.report['hardware']['camera']['status'] = 'error'
            self.report['hardware']['camera']['error'] = str(e)
            self._check('camera', False, f"摄像头检查出错: {e}")
            print(f"✗ 摄像头检查出错: {e}")
            return False

    def check_microphone(self):
        print("\n" + "="*60)
        print("检查麦克风可用性...")
        print("="*60)
        
        self.report['hardware']['microphone'] = {}
        devices = []
        
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            device_count = p.get_device_count()
            default_input = p.get_default_input_device_info()
            
            for i in range(device_count):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'default_sample_rate': int(device_info['defaultSampleRate'])
                    })
            
            p.terminate()
            
            self.report['hardware']['microphone']['devices'] = devices
            self.report['hardware']['microphone']['default'] = {
                'name': default_input['name'],
                'index': default_input['index']
            }
            
            if devices:
                self.report['hardware']['microphone']['status'] = 'available'
                print(f"✓ 发现 {len(devices)} 个麦克风设备:")
                for dev in devices:
                    print(f"  - [{dev['index']}] {dev['name']} ({dev['channels']}通道)")
                print(f"  默认麦克风: {default_input['name']}")
                self._check('microphone', True)
                return True
            else:
                self.report['hardware']['microphone']['status'] = 'no_devices'
                self._check('microphone', False, "未发现可用的麦克风设备")
                print("✗ 未发现可用的麦克风设备")
                return False
                
        except ImportError:
            self.report['hardware']['microphone']['status'] = 'pyaudio_missing'
            self._check('microphone', False, "pyaudio 未安装，无法检查麦克风", warning=True)
            print("⚠ pyaudio 未安装，无法检查麦克风")
            return False
        except Exception as e:
            self.report['hardware']['microphone']['status'] = 'error'
            self.report['hardware']['microphone']['error'] = str(e)
            self._check('microphone', False, f"麦克风检查出错: {e}")
            print(f"✗ 麦克风检查出错: {e}")
            return False

    def check_speaker(self):
        print("\n" + "="*60)
        print("检查扬声器可用性...")
        print("="*60)
        
        self.report['hardware']['speaker'] = {}
        
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            device_count = p.get_device_count()
            devices = []
            
            for i in range(device_count):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxOutputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxOutputChannels']
                    })
            
            p.terminate()
            
            self.report['hardware']['speaker']['devices'] = devices
            self.report['hardware']['speaker']['platform'] = platform.system()
            
            if devices:
                self.report['hardware']['speaker']['status'] = 'available'
                print(f"✓ 发现 {len(devices)} 个音频输出设备:")
                for dev in devices:
                    print(f"  - [{dev['index']}] {dev['name']} ({dev['channels']}通道)")
                self._check('speaker', True)
                return True
            else:
                self.report['hardware']['speaker']['status'] = 'no_devices'
                self._check('speaker', False, "未发现可用的音频输出设备")
                print("✗ 未发现可用的音频输出设备")
                return False
                
        except ImportError:
            self.report['hardware']['speaker']['status'] = 'pyaudio_missing'
            self._check('speaker', False, "pyaudio 未安装，无法检查扬声器", warning=True)
            print("⚠ pyaudio 未安装，无法检查扬声器")
            return False
        except Exception as e:
            self.report['hardware']['speaker']['status'] = 'error'
            self.report['hardware']['speaker']['error'] = str(e)
            self._check('speaker', False, f"扬声器检查出错: {e}")
            print(f"✗ 扬声器检查出错: {e}")
            return False

    def check_robot_arm(self):
        print("\n" + "="*60)
        print("检查机械臂连接状态...")
        print("="*60)
        
        self.report['robot_arm'] = {}
        system = platform.system()
        
        if system != 'Linux':
            self.report['robot_arm']['status'] = 'unsupported_platform'
            self.report['robot_arm']['platform'] = system
            self._check('robot_arm', False, f"机械臂仅支持 Linux (树莓派)，当前平台: {system}", warning=True)
            print(f"⚠ 机械臂仅支持 Linux (树莓派)，当前平台: {system}")
            return False
        
        try:
            from pymycobot.mycobot import MyCobot
            from pymycobot import PI_PORT, PI_BAUD
            
            self.report['robot_arm']['library'] = 'available'
            
            try:
                mc = MyCobot(PI_PORT, PI_BAUD)
                
                angles = mc.get_angles()
                coords = mc.get_coords()
                is_power_on = mc.is_power_on()
                
                self.report['robot_arm']['connected'] = True
                self.report['robot_arm']['power_on'] = is_power_on
                self.report['robot_arm']['angles'] = angles
                self.report['robot_arm']['coords'] = coords
                
                print(f"✓ 机械臂连接成功")
                print(f"  电机状态: {'上电' if is_power_on else '下电'}")
                print(f"  当前角度: {angles}")
                self._check('robot_arm', True)
                
                try:
                    import RPi.GPIO as GPIO
                    self.report['robot_arm']['gpio_available'] = True
                    print(f"  GPIO 库: 可用")
                except ImportError:
                    self.report['robot_arm']['gpio_available'] = False
                    self._check('gpio', False, "RPi.GPIO 未安装", warning=True)
                    print(f"  ⚠ RPi.GPIO 未安装")
                
                return True
                
            except Exception as e:
                self.report['robot_arm']['connected'] = False
                self.report['robot_arm']['error'] = str(e)
                self._check('robot_arm', False, f"机械臂连接失败: {e}")
                print(f"✗ 机械臂连接失败: {e}")
                return False
                
        except ImportError as e:
            self.report['robot_arm']['library'] = 'missing'
            self.report['robot_arm']['error'] = str(e)
            self._check('robot_arm', False, "pymycobot 未安装")
            print("✗ pymycobot 未安装")
            return False

    def _is_placeholder(self, value):
        if not isinstance(value, str):
            return False
        value = value.strip()
        for pattern in PLACEHOLDER_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False

    def _extract_key_values(self, content):
        key_values = {}
        
        for key in REQUIRED_KEYS:
            pattern = rf'^{key}\s*=\s*["\']([^"\']*)["\']'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                key_values[key] = match.group(1)
            else:
                key_values[key] = None
        
        return key_values

    def check_config(self):
        print("\n" + "="*60)
        print("检查配置文件...")
        print("="*60)
        
        self.report['config'] = {
            'template_path': str(self.template_path),
            'target_path': str(self.config_path)
        }
        
        template_exists = self.template_path.exists()
        config_exists = self.config_path.exists()
        
        self.report['config']['template_exists'] = template_exists
        self.report['config']['config_exists'] = config_exists
        
        if not template_exists:
            print(f"⚠  模板文件不存在: {self.template_path}")
            print(f"  请运行 python env_check.py --fix 生成配置模板")
        
        if not config_exists:
            self.report['config']['status'] = 'missing'
            self._check('config', False, f"配置文件不存在: {self.config_path}")
            print(f"✗ 配置文件不存在: {self.config_path}")
            print(f"  请运行 python env_check.py --fix 创建配置文件")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            key_values = self._extract_key_values(content)
            
            found_keys = {}
            placeholder_keys = []
            configured_keys = []
            
            for key, value in key_values.items():
                if value is not None:
                    found_keys[key] = True
                    if self._is_placeholder(value):
                        placeholder_keys.append(key)
                    else:
                        configured_keys.append(key)
                else:
                    found_keys[key] = False
            
            self.report['config']['keys_found'] = found_keys
            self.report['config']['placeholder_keys'] = placeholder_keys
            self.report['config']['configured_keys'] = configured_keys
            self.report['config']['status'] = 'exists'
            
            print(f"✓ 配置文件存在: {self.config_path}")
            print(f"  检测到的配置项:")
            
            for key, found in found_keys.items():
                if found:
                    if key in placeholder_keys:
                        print(f"    ⚠ {key}: 使用占位符 (需要配置真实密钥)")
                    else:
                        print(f"    ✓ {key}: 已配置")
                else:
                    print(f"    ✗ {key}: 缺失")
            
            all_found = all(found_keys.values())
            any_placeholder = len(placeholder_keys) > 0
            
            if all_found and not any_placeholder:
                self._check('config', True)
                print(f"\n✓ 所有配置项已正确配置")
            elif all_found and any_placeholder:
                self._check('config', False, f"仍有 {len(placeholder_keys)} 个配置项使用占位符")
                print(f"\n⚠  请将占位符替换为真实的 API 密钥")
                for key in placeholder_keys:
                    print(f"   - {key}")
            else:
                self._check('config', False, "部分配置项缺失")
            
            self.report['config']['valid'] = all_found and not any_placeholder
            return self.report['config']['valid']
            
        except Exception as e:
            self.report['config']['status'] = 'error'
            self.report['config']['error'] = str(e)
            self._check('config', False, f"配置文件读取错误: {e}")
            print(f"✗ 配置文件读取错误: {e}")
            return False

    def generate_config_template(self):
        print("\n" + "="*60)
        print("生成配置模板...")
        print("="*60)
        
        template_content = '''# API_KEY.py
# 同济子豪兄 2024-5-22
# 各种开放平台的KEY，不要外传
#
# 配置说明：
# 1. 从各开放平台获取真实的 API Key
# 2. 将下方的占位符替换为真实密钥
# 3. 不要将包含真实密钥的文件提交到版本控制系统
# 4. 此文件中的默认占位符包括: your_*_here, XXX, *** 等

# 通义千问QwenVL系列
# https://bailian.console.aliyun.com/#/model-market
Qwen_KEY = "your_qwen_key_here"


# 零一万物大模型开放平台
# https://platform.lingyiwanwu.com
YI_KEY = "your_yi_key_here"


# 百度智能云千帆ModelBuilder
# https://qianfan.cloud.baidu.com
QIANFAN_ACCESS_KEY = "your_qianfan_access_key_here"
QIANFAN_SECRET_KEY = "your_qianfan_secret_key_here"

# 百度智能云千帆AppBuilder-SDK
APPBUILDER_TOKEN = "your_appbuilder_token_here"
'''
        
        self.template_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        print(f"✓ 配置模板已生成: {self.template_path}")
        
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.template_path, self.config_path)
            print(f"✓ 已创建配置文件: {self.config_path}")
            print("  请编辑该文件，填入您的真实 API 密钥")
        else:
            print(f"\n  配置文件已存在: {self.config_path}")
            print("  如需重置配置，请删除该文件后重新运行 --fix")
        
        return self.template_path

    def run_all_checks(self):
        print("\n" + "#"*60)
        print("# 环境自检与配置向导")
        print("#"*60)
        
        self.check_python_version()
        self.check_dependencies()
        self.check_camera()
        self.check_microphone()
        self.check_speaker()
        self.check_robot_arm()
        self.check_config()
        
        self.print_summary()
        self.save_report()
        
        return self.report['summary']['failed'] == 0

    def print_summary(self):
        print("\n" + "#"*60)
        print("# 诊断摘要")
        print("#"*60)
        
        summary = self.report['summary']
        print(f"\n总检查项: {summary['total_checks']}")
        print(f"  ✓ 通过: {summary['passed']}")
        print(f"  ⚠ 警告: {summary['warnings']}")
        print(f"  ✗ 失败: {summary['failed']}")
        
        if self._issues:
            print("\n检测到的问题:")
            for issue in self._issues:
                print(f"  {issue}")
        
        if summary['failed'] == 0:
            print("\n✓ 所有检查通过！环境配置完整。")
        else:
            print("\n✗ 部分检查未通过，请根据上面的提示修复问题。")
            print("  运行 python env_check.py --fix 可自动生成配置模板。")

    def save_report(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.project_root / f'env_report_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 诊断报告已保存: {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(
        description='环境自检与配置向导',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
版本要求: Python >= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}

示例:
  python env_check.py                    # 运行所有检查
  python env_check.py --fix              # 生成配置模板
  python env_check.py --config path/to/API_KEY.py  # 指定配置文件路径
        '''
    )
    parser.add_argument('--fix', action='store_true', 
                       help='自动生成/更新配置模板')
    parser.add_argument('--config', type=str, default=None,
                       help='指定配置文件路径 (默认: agent_demo_20250328/API_KEY.py)')
    parser.add_argument('--report', type=str, default=None,
                       help='指定报告输出路径 (JSON格式)')
    
    args = parser.parse_args()
    
    checker = EnvironmentChecker(config_path=args.config)
    
    if args.fix:
        checker.generate_config_template()
        print("\n运行 python env_check.py 进行完整环境检查")
        return
    
    success = checker.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
