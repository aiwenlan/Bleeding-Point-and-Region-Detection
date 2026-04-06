#!/usr/bin/env python3
"""
BlooDet预训练模型下载脚本
自动下载SAM2和PWC-Net的预训练权重
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path
import zipfile
import tarfile

# 模型下载配置
MODELS = {
    'sam2_base': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
        'filename': 'sam2_base.pth',
        'md5': None,  # 实际使用时需要填入正确的MD5
        'description': 'SAM2 Base+ model for image encoding'
    },
    'sam2_large': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',
        'filename': 'sam2_large.pth', 
        'md5': None,
        'description': 'SAM2 Large model for image encoding'
    },
    'pwcnet_chairs': {
        'url': 'https://github.com/NVlabs/PWC-Net/raw/master/PyTorch/pwc_net_chairs.pth.tar',
        'filename': 'pwcnet_chairs.pth',
        'md5': None,
        'description': 'PWC-Net trained on FlyingChairs dataset'
    },
    'pwcnet_things': {
        'url': 'https://github.com/NVlabs/PWC-Net/raw/master/PyTorch/pwc_net.pth.tar',
        'filename': 'pwcnet_things.pth',
        'md5': None,
        'description': 'PWC-Net trained on FlyingThings3D dataset'
    }
}

def calculate_md5(filepath):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, filepath, description=""):
    """下载文件并显示进度"""
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"Destination: {filepath}")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            print(f"\rProgress: {percent}% [{block_num * block_size}/{total_size} bytes]", end='')
        else:
            print(f"\rDownloaded: {block_num * block_size} bytes", end='')
    
    try:
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

def verify_file(filepath, expected_md5):
    """验证文件完整性"""
    if expected_md5 is None:
        print("No MD5 checksum provided, skipping verification")
        return True
    
    print("Verifying file integrity...")
    actual_md5 = calculate_md5(filepath)
    
    if actual_md5 == expected_md5:
        print("✓ File verification passed")
        return True
    else:
        print(f"✗ File verification failed!")
        print(f"Expected MD5: {expected_md5}")
        print(f"Actual MD5:   {actual_md5}")
        return False

def download_model(model_name, checkpoints_dir, force_download=False):
    """下载指定模型"""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        return False
    
    model_info = MODELS[model_name]
    filepath = checkpoints_dir / model_info['filename']
    
    # 检查文件是否已存在
    if filepath.exists() and not force_download:
        print(f"Model {model_name} already exists at {filepath}")
        if verify_file(filepath, model_info['md5']):
            return True
        else:
            print("Existing file is corrupted, re-downloading...")
    
    # 下载文件
    success = download_file(model_info['url'], filepath, model_info['description'])
    
    if success:
        # 验证文件
        if verify_file(filepath, model_info['md5']):
            print(f"✓ Successfully downloaded {model_name}")
            return True
        else:
            # 删除损坏的文件
            filepath.unlink()
            return False
    
    return False

def setup_pwcnet_dependencies():
    """设置PWC-Net依赖"""
    print("\nSetting up PWC-Net dependencies...")
    
    # 检查correlation package
    try:
        import correlation_package
        print("✓ Correlation package already available")
        return True
    except ImportError:
        print("Correlation package not found")
        
        # 尝试编译correlation package
        pwc_dir = Path(__file__).parent.parent / "PWC-Net" / "PyTorch"
        corr_dir = pwc_dir / "external_packages" / "correlation-pytorch-master"
        
        if corr_dir.exists():
            print(f"Found correlation package source at {corr_dir}")
            print("Please manually compile the correlation package:")
            print(f"cd {corr_dir}")
            print("python setup.py install")
            return False
        else:
            print("Correlation package source not found")
            print("Please check PWC-Net directory structure")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download BlooDet pretrained models')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()) + ['all'],
                       default=['sam2_base', 'pwcnet_things'],
                       help='Models to download (default: sam2_base, pwcnet_things)')
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints',
                       help='Directory to save models (default: ./checkpoints)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')
    parser.add_argument('--setup-deps', action='store_true',
                       help='Setup PWC-Net dependencies')
    
    args = parser.parse_args()
    
    # 创建checkpoints目录
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"BlooDet Model Downloader")
    print(f"Checkpoints directory: {checkpoints_dir.absolute()}")
    print("=" * 60)
    
    # 确定要下载的模型
    if 'all' in args.models:
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = args.models
    
    # 下载模型
    success_count = 0
    total_count = len(models_to_download)
    
    for model_name in models_to_download:
        print(f"\n[{success_count + 1}/{total_count}] Processing {model_name}...")
        if download_model(model_name, checkpoints_dir, args.force):
            success_count += 1
        else:
            print(f"✗ Failed to download {model_name}")
    
    # 设置依赖
    if args.setup_deps:
        setup_pwcnet_dependencies()
    
    # 总结
    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count}/{total_count} models downloaded successfully")
    
    if success_count == total_count:
        print("✓ All models downloaded successfully!")
        print("\nNext steps:")
        print("1. Update your config file to point to the downloaded models")
        print("2. If using PWC-Net, compile the correlation package (see --setup-deps)")
        print("3. Start training with: python train.py")
    else:
        print("⚠ Some downloads failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
