"""
Batch Test Script for PiX

This script automatically:
1. Finds all .npz files in the data directory
2. Updates the dataset_path in config.yaml for each file
3. Runs single_test.py with a specific hypothesis combination
4. Collects and reports the results
"""
import os
import sys
import glob
import subprocess
import time
from pathlib import Path
from omegaconf import OmegaConf
import shutil

# 项目根目录
PROJECT_ROOT = "/Users/oscar/Desktop/OX/CS/AI/PDE/Code/PiX"
DATA_DIR = os.path.join(PROJECT_ROOT, "pix", "data")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "pix", "cfg", "config.yaml")
SINGLE_TEST_PATH = os.path.join(PROJECT_ROOT, "pix", "single_test.py")

# 要测试的假设组合
HYPOTHESIS_COMBINATION = [0, 1, 2, 4, 6, 7, 8, 10, 14, 15, 32]

def backup_config():
    """
    备份原始配置文件
    
    Returns:
        str: 备份文件路径
    """
    backup_path = CONFIG_PATH + ".backup"
    shutil.copy2(CONFIG_PATH, backup_path)
    print(f"配置文件已备份到: {backup_path}")
    return backup_path

def restore_config(backup_path):
    """
    恢复原始配置文件
    
    Args:
        backup_path: 备份文件路径
    """
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, CONFIG_PATH)
        os.remove(backup_path)
        print(f"配置文件已恢复")
    else:
        print(f"警告: 备份文件不存在: {backup_path}")

def update_config_dataset_path(dataset_path):
    """
    更新配置文件中的 dataset_path
    
    Args:
        dataset_path: 新的数据集路径
    """
    try:
        # 读取配置文件
        cfg = OmegaConf.load(CONFIG_PATH)
        
        # 更新 dataset_path（使用相对路径）
        relative_path = os.path.relpath(dataset_path, PROJECT_ROOT)
        cfg.dataset_path = relative_path
        
        # 保存配置文件
        OmegaConf.save(cfg, CONFIG_PATH)
        print(f"已更新配置文件，dataset_path = {relative_path}")
        
    except Exception as e:
        print(f"更新配置文件失败: {e}")
        raise

def find_npz_files():
    """
    查找数据目录中的所有 .npz 文件
    
    Returns:
        list: npz 文件路径列表
    """
    pattern = os.path.join(DATA_DIR, "*.npz")
    npz_files = glob.glob(pattern)
    
    # 过滤掉隐藏文件和其他不需要的文件
    npz_files = [f for f in npz_files if not os.path.basename(f).startswith('.')]
    
    print(f"找到 {len(npz_files)} 个 .npz 文件:")
    for file in npz_files:
        print(f"  - {os.path.basename(file)}")
    
    return npz_files

def run_single_test(hypothesis_combination):
    """
    运行 single_test.py 并获取结果
    
    Args:
        hypothesis_combination: 假设编号组合列表
        
    Returns:
        dict: 包含测试结果的字典，如果失败返回None
    """
    try:
        # 准备命令行参数
        hypothesis_str = ",".join(map(str, hypothesis_combination))
        
        # 运行 single_test.py
        print(f"运行测试，假设组合: {hypothesis_combination}")
        cmd = [sys.executable, SINGLE_TEST_PATH, hypothesis_str]
        
        # 切换到项目根目录执行
        result = subprocess.run(
            cmd, 
            cwd=PROJECT_ROOT, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            # 解析输出，提取损失值
            output = result.stdout
            train_loss = None
            valid_loss = None
            
            # 查找训练损失和验证损失
            for line in output.split('\n'):
                if '训练损失:' in line:
                    try:
                        train_loss = float(line.split('训练损失:')[1].strip())
                    except:
                        pass
                elif '验证损失:' in line:
                    try:
                        valid_loss = float(line.split('验证损失:')[1].strip())
                    except:
                        pass
            
            return {
                'success': True,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'output': output,
                'error': None
            }
        else:
            return {
                'success': False,
                'train_loss': None,
                'valid_loss': None,
                'output': result.stdout,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'train_loss': None,
            'valid_loss': None,
            'output': None,
            'error': 'Timeout (>5 minutes)'
        }
    except Exception as e:
        return {
            'success': False,
            'train_loss': None,
            'valid_loss': None,
            'output': None,
            'error': str(e)
        }

def main():
    """
    主函数
    """
    print("=== PiX 批量测试脚本 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"假设组合: {HYPOTHESIS_COMBINATION}")
    print()
    
    # 检查必要文件是否存在
    if not os.path.exists(CONFIG_PATH):
        print(f"错误: 配置文件不存在: {CONFIG_PATH}")
        return
    
    if not os.path.exists(SINGLE_TEST_PATH):
        print(f"错误: single_test.py 不存在: {SINGLE_TEST_PATH}")
        return
    
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录不存在: {DATA_DIR}")
        return
    
    # 查找所有 npz 文件
    npz_files = find_npz_files()
    if not npz_files:
        print("错误: 没有找到任何 .npz 文件")
        return
    
    # 备份原始配置
    backup_path = backup_config()
    
    # 结果存储
    results = []
    
    try:
        print("\n=== 开始批量测试 ===")
        
        for i, npz_file in enumerate(npz_files, 1):
            filename = os.path.basename(npz_file)
            print(f"\n[{i}/{len(npz_files)}] 测试文件: {filename}")
            print("-" * 50)
            
            try:
                # 更新配置文件
                update_config_dataset_path(npz_file)
                
                # 运行测试
                start_time = time.time()
                test_result = run_single_test(HYPOTHESIS_COMBINATION)
                end_time = time.time()
                
                # 记录结果
                result_entry = {
                    'file': filename,
                    'file_path': npz_file,
                    'hypothesis_combination': HYPOTHESIS_COMBINATION.copy(),
                    'success': test_result['success'],
                    'train_loss': test_result['train_loss'],
                    'valid_loss': test_result['valid_loss'],
                    'duration': end_time - start_time,
                    'error': test_result['error']
                }
                results.append(result_entry)
                
                # 显示结果
                if test_result['success']:
                    print(f"✓ 测试成功")
                    if test_result['train_loss'] is not None:
                        print(f"  训练损失: {test_result['train_loss']:.6f}")
                    if test_result['valid_loss'] is not None:
                        print(f"  验证损失: {test_result['valid_loss']:.6f}")
                    print(f"  用时: {result_entry['duration']:.1f} 秒")
                else:
                    print(f"✗ 测试失败")
                    if test_result['error']:
                        print(f"  错误: {test_result['error']}")
                
            except Exception as e:
                print(f"✗ 处理文件时出错: {e}")
                result_entry = {
                    'file': filename,
                    'file_path': npz_file,
                    'hypothesis_combination': HYPOTHESIS_COMBINATION.copy(),
                    'success': False,
                    'train_loss': None,
                    'valid_loss': None,
                    'duration': 0,
                    'error': str(e)
                }
                results.append(result_entry)
    
    finally:
        # 恢复原始配置
        restore_config(backup_path)
    
    # 显示汇总结果
    print("\n" + "=" * 60)
    print("=== 批量测试结果汇总 ===")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"总文件数: {len(results)}")
    print(f"成功测试: {success_count}")
    print(f"失败测试: {len(results) - success_count}")
    print()
    
    # 按照损失值排序（成功的测试）
    successful_results = [r for r in results if r['success'] and r['valid_loss'] is not None]
    successful_results.sort(key=lambda x: x['valid_loss'])
    
    if successful_results:
        print("成功测试结果 (按验证损失排序):")
        print(f"{'文件名':<30} {'训练损失':<12} {'验证损失':<12} {'用时(秒)':<8}")
        print("-" * 70)
        for result in successful_results:
            print(f"{result['file']:<30} {result['train_loss']:<12.6f} {result['valid_loss']:<12.6f} {result['duration']:<8.1f}")
    
    # 显示失败的测试
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n失败测试详情:")
        for result in failed_results:
            print(f"  {result['file']}: {result['error']}")
    
    print("\n=== 批量测试完成 ===")

if __name__ == "__main__":
    main()
