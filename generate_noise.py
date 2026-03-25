# -*- coding: utf-8 -*-
"""
纯噪声数据集生成脚本
用途：生成不含任何流星信号的纯噪声数据。
逻辑：输入为高斯白噪声，对应的 Clean 标签为全 0。
这将帮助 DCU-Net 学习抑制背景噪声，防止产生虚假信号。
"""
import numpy as np
import h5py
import os
import pandas as pd

def generate_pure_noise(output_folder, num_samples=3000, signal_length=440):
    """
    生成纯噪声数据并保存为 h5 和 txt
    """
    os.makedirs(output_folder, exist_ok=True)
    
    h5_filename = os.path.join(output_folder, 'noise_only.h5')
    txt_filename = os.path.join(output_folder, 'noise_only.txt')
    
    print(f'正在生成 {num_samples} 条纯噪声数据...')
    
    # 1. 生成复高斯白噪声 (Circularly Symmetric Complex Gaussian Noise)
    # 实部和虚部均服从 N(0, 1)
    noise_real = np.random.randn(num_samples, signal_length)
    noise_imag = np.random.randn(num_samples, signal_length)
    noise_signals = noise_real + 1j * noise_imag
    
    # 2. 归一化
    # 即使是纯噪声，进入网络前也通常需要归一化到 [-1, 1] 范围
    # 这里按每条数据的最大幅值进行归一化，模拟接收机增益控制（AGC）后的效果
    max_vals = np.max(np.abs(noise_signals), axis=1, keepdims=True)
    # 防止除以0
    max_vals[max_vals == 0] = 1.0
    
    # 输入数据：归一化后的噪声
    noisy_input = noise_signals / max_vals
    
    # 标签数据：全 0 (因为没有流星)
    # 注意：在训练加载器中，通常会读取 snr_inf.h5 作为标签。
    # 对于纯噪声，我们需要一种机制告诉 DataLoader 标签是全0。
    # 这里我们生成一个特殊的 h5 文件，内部结构包含 noisy 和 clean (全0) 
    # 或者，你可以将此文件命名为 snr_-999.h5，并依靠后续逻辑处理。
    # 为了兼容之前的 load_h5_signals 结构，这里只存一份数据，但在训练时需要特殊处理，
    # 或者我们直接在这里生成两个 dataset：'signals_real' (噪声) 和 'clean_real' (全0)
    # 但为了保持h5结构完全一致（只存一组数据），我们只存噪声。
    # *** 关键策略 ***：在 Dataset 类中加载此文件时，需要知道对应的 Clean 是 0。
    
    # 3. 保存 h5 文件
    print(f'保存到: {h5_filename}')
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('signals_real', data=noisy_input.real, dtype=np.float64)
        f.create_dataset('signals_imag', data=noisy_input.imag, dtype=np.float64)
        
        # 保存一些属性
        f.attrs['num_meteors'] = num_samples
        f.attrs['signal_length'] = signal_length
        f.attrs['snr_dB'] = 'noise_only' # 特殊标记
        f.attrs['dtype'] = 'complex128'

    # 4. 生成元数据 txt (为了保持格式一致，填入 dummy 数据)
    # 获取之前的列名（硬编码以匹配之前的脚本）
    columns = ['idx', 'SNR', 'TSd', 'Est', 'Rg', 'Err', 'SNR_csv', 'Pwr', 
               'Aoa_a', 'Aoa_z', 'Td', 'ETd', 'Dc', 'EDc', 'Vr', 'EVr', 
               'Vm', 'EVm', 'Rgf', 'Rga']
    
    print(f'保存元数据到: {txt_filename}')
    with open(txt_filename, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write(','.join(columns) + '\n')
        
        for i in range(num_samples):
            # 构建一行全为 0 或 NaN 的数据
            # idx, SNR, ...others
            line = [str(i), 'noise_only'] + ['0'] * (len(columns) - 2)
            f.write(','.join(line) + '\n')

    print('✓ 纯噪声数据集生成完成！')

if __name__ == "__main__":
    output_folder = r'F:\denoise'
    # 生成 3000 条纯噪声，与其他 SNR 数量保持一致
    generate_pure_noise(output_folder, num_samples=3000)