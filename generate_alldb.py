# -*- coding: utf-8 -*-
"""
完美批量生成脚本 (Perfect Batch Generator)
目标：
1. 保证 100% 样本有效性：生成直到凑满 6000 个有效 Clean 信号。
2. 保证 100% 对齐：所有 SNR 数据均由这 6000 个 Clean 信号数学加噪生成。
3. 保证 100% 速度：物理仿真只跑 Clean，其余全部向量化加噪。
"""
import numpy as np
import pandas as pd
import h5py
import os
import glob
from multiprocessing import Pool, cpu_count
import time
import warnings
from h_simulation_py import simulate_single_meteor

warnings.filterwarnings('ignore')

# ==========================================
# 配置
# ==========================================
CSV_FOLDER = r'CSV202502'       
OUTPUT_FOLDER = r'F:\denoise_perfect' # 建议新建文件夹
TARGET_COUNT = 6000             # 必须凑满的数量
MAX_RETRIES = 3                 # 单个样本重试次数

# 目标 SNR 列表
TARGET_SNRS = [
    -10, 0, 3, 5, 6, 7, 8, 9, 10, 
    11, 12, 13, 14, 15, 20, 25, 30, 
    35, 40
]

# ==========================================
# 1. 基础物理仿真 (Clean Generator)
# ==========================================
def load_csv_files(csv_folder):
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    all_meteors = []
    print(f'>>> 读取 CSV 文件...')
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if 'Err' not in df.columns and 'err' in df.columns: df['Err'] = df['err']
            if 'Err' in df.columns:
                df_clean = df[df['Err'] == 0].copy()
                if len(df_clean) > 0: all_meteors.append(df_clean)
        except: pass
    if not all_meteors: return None
    return pd.concat(all_meteors, ignore_index=True)

def map_params(row):
    def get(name):
        for c in row.index:
            if c.lower() == name.lower(): return row[c]
        return None
    params = {
        'Theta_z': float(get('Aoa_z') or 64.0),
        'Theta_a': float(get('Aoa_a') or 132.0),
        'v': float(get('Vm') or 37548.0),
        'R0': float(get('Rg') or 219.0) * 1000,
        'V': float(get('Vr') or 11.0),
        'q': 1e14
    }
    pwr = get('Pwr')
    if pwr and pwr > 0: params['q'] = max(1e12, min(1e16, pwr / 1e10))
    return params

def worker_try_generate_clean(args):
    """
    尝试生成一个 Clean 信号
    返回: (signal, row_data, True/False)
    """
    row_dict, max_retries = args
    row = pd.Series(row_dict)
    params = map_params(row)
    
    for _ in range(max_retries):
        try:
            sig = simulate_single_meteor(
                SNR_dB=np.inf, # 始终生成纯净信号
                Theta_z=params['Theta_z'],
                Theta_a=params['Theta_a'],
                v=params['v'],
                R0=params['R0'],
                V=params['V'],
                q=params['q']
            )
            # 严格检查有效性
            if sig is not None and len(sig) == 440:
                if np.max(np.abs(sig)) > 1e-12:
                    return (sig, row_dict, True)
        except: pass
    
    return (None, None, False)

# ==========================================
# 2. 快速加噪 (Vectorized)
# ==========================================
def add_noise_vectorized(clean_signals, target_snr_db):
    """
    数学加噪 (Peak SNR 版本)
    定义：SNR = P_peak / P_noise
    """
    n_samples, length = clean_signals.shape
    
    # === 修改点：使用 Max (峰值功率) 而不是 Mean ===
    # p_peak: 每个信号的最大功率 [N,]
    p_peak = np.max(np.abs(clean_signals)**2, axis=1)
    
    # 转换 SNR dB -> 线性比例
    snr_linear = 10 ** (target_snr_db / 10.0)
    
    # 计算噪声功率：让噪声的方差适应信号的峰值
    # P_noise = P_peak / SNR
    p_noise = p_peak / (snr_linear + 1e-20)
    
    # 生成复高斯白噪声 CN(0, 1) -> 初始功率为 1 (实部0.5 + 虚部0.5)
    raw_noise = (np.random.randn(n_samples, length) + 1j * np.random.randn(n_samples, length))
    
    # 调整噪声幅度
    # 目标：噪声的功率 (方差) 应该是 p_noise
    # raw_noise / sqrt(2) 具有单位功率 1
    # 乘以 sqrt(p_noise) 得到目标功率
    
    scale = np.sqrt(p_noise)[:, np.newaxis] # [N, 1] 广播
    
    # 生成最终噪声
    # 注意：这里噪声幅度会比之前大很多！
    noise = (raw_noise / np.sqrt(2)) * scale
    
    # 叠加
    noisy_signals = clean_signals + noise
    
    return noisy_signals

# ==========================================
# 主流程
# ==========================================
def main():
    start_time = time.time()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. 读取 CSV 数据池
    df_pool = load_csv_files(CSV_FOLDER)
    if df_pool is None: return
    print(f"数据池大小: {len(df_pool)} 条参数")
    
    # 2. 凑满 TARGET_COUNT 个 Clean 信号
    print(f"\n>>> [Step 1] 正在筛选并生成 {TARGET_COUNT} 个黄金 Clean 信号...")
    
    valid_clean_signals = []
    valid_metadata = []
    
    num_cores = max(1, cpu_count() - 2)
    pool = Pool(num_cores)
    
    # 批量大小，每次多取一点以防失败
    batch_size = int(TARGET_COUNT * 1.2) 
    
    while len(valid_clean_signals) < TARGET_COUNT:
        needed = TARGET_COUNT - len(valid_clean_signals)
        print(f"  当前进度: {len(valid_clean_signals)}/{TARGET_COUNT} (还差 {needed} 个)")
        
        # 从池中随机抽样
        sample_df = df_pool.sample(n=min(len(df_pool), int(needed * 1.5)), replace=True)
        tasks = [(row, MAX_RETRIES) for row in sample_df.to_dict('records')]
        
        results = pool.map(worker_try_generate_clean, tasks)
        
        for sig, meta, success in results:
            if success:
                valid_clean_signals.append(sig)
                valid_metadata.append(meta)
                if len(valid_clean_signals) >= TARGET_COUNT:
                    break
    
    pool.close()
    pool.join()
    
    print(f"  >>> 成功凑齐 {TARGET_COUNT} 个有效信号！")
    
    # 转为 numpy 矩阵 [6000, 440]
    clean_matrix = np.array(valid_clean_signals, dtype=complex)
    
    # 保存 Master Metadata (只有这 6000 个有效的)
    meta_df = pd.DataFrame(valid_metadata)
    meta_df['Index_In_File'] = range(TARGET_COUNT)
    meta_df.to_csv(os.path.join(OUTPUT_FOLDER, 'master_metadata.csv'), index=False)
    print("  >>> Metadata 已保存")

    # 保存 Clean H5
    with h5py.File(os.path.join(OUTPUT_FOLDER, 'snr_inf.h5'), 'w') as f:
        f.create_dataset('signals_real', data=clean_matrix.real)
        f.create_dataset('signals_imag', data=clean_matrix.imag)
        f.attrs['snr'] = 'inf'
        f.attrs['count'] = TARGET_COUNT
    
    # 3. 快速生成其他 SNR (加噪)
    print(f"\n>>> [Step 2] 极速衍生含噪数据...")
    
    for snr in TARGET_SNRS:
        t0 = time.time()
        print(f"  Generating SNR = {snr} dB ... ", end='')
        
        # 矩阵加噪 (0.1秒内完成)
        noisy_matrix = add_noise_vectorized(clean_matrix, snr)
        
        # 保存
        fname = f'snr_{int(snr)}.h5'
        with h5py.File(os.path.join(OUTPUT_FOLDER, fname), 'w') as f:
            f.create_dataset('signals_real', data=noisy_matrix.real)
            f.create_dataset('signals_imag', data=noisy_matrix.imag)
            f.attrs['snr'] = str(int(snr))
            f.attrs['count'] = TARGET_COUNT
            
        print(f"Done ({time.time()-t0:.2f}s)")

    # 4. 生成纯噪声
    print(f"\n>>> [Step 3] 生成纯噪声...")
    noise_only = (np.random.randn(TARGET_COUNT, 440) + 1j*np.random.randn(TARGET_COUNT, 440))
    # 归一化
    m = np.max(np.abs(noise_only), axis=1, keepdims=True)
    noise_only /= (m + 1e-9)
    
    with h5py.File(os.path.join(OUTPUT_FOLDER, 'noise_only.h5'), 'w') as f:
        f.create_dataset('signals_real', data=noise_only.real)
        f.create_dataset('signals_imag', data=noise_only.imag)
        f.attrs['snr'] = 'noise'
        f.attrs['count'] = TARGET_COUNT

    print(f"\n=== 全部完成，总耗时: {(time.time()-start_time)/60:.1f} 分钟 ===")
    print(f"数据路径: {OUTPUT_FOLDER}")
    print("此数据集保证：1.无空缺 2.绝对对齐。可以直接训练。")

if __name__ == '__main__':
    main()