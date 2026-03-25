# -*- coding: utf-8 -*-
"""
可视化脚本：从h5文件中随机选择信号并绘制
"""
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib字体和负号显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_h5_signals(h5_filename):
    """读取h5文件中的信号数据"""
    with h5py.File(h5_filename, 'r') as f:
        signals_real = f['signals_real'][:]
        signals_imag = f['signals_imag'][:]
        signals = signals_real + 1j * signals_imag
        num_meteors = f.attrs['num_meteors']
        signal_length = f.attrs['signal_length']
        snr_dB = f.attrs['snr_dB']
    return signals, num_meteors, signal_length, snr_dB

def load_metadata(txt_filename):
    """读取txt元数据文件"""
    try:
        df = pd.read_csv(txt_filename)
        return df
    except Exception as e:
        print(f'读取元数据失败: {e}')
        return None

def visualize_random_signals(h5_folder, num_samples_per_snr=3, random_seed=42):
    """
    从每个SNR的h5文件中随机选择信号并可视化
    
    参数:
        h5_folder: h5文件所在文件夹
        num_samples_per_snr: 每个SNR随机选择的信号数量
        random_seed: 随机种子
    """
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 查找所有h5文件
    h5_files = sorted(glob.glob(os.path.join(h5_folder, 'snr_*.h5')))
    
    if len(h5_files) == 0:
        print(f'在 {h5_folder} 中没有找到h5文件！')
        return
    
    print(f'找到 {len(h5_files)} 个h5文件')
    
    # 为每个SNR创建子图
    num_snrs = len(h5_files)
    fig, axes = plt.subplots(num_snrs, num_samples_per_snr, 
                             figsize=(6*num_samples_per_snr, 4.5*num_snrs))
    
    # 如果只有一个SNR，确保axes是二维数组
    if num_snrs == 1:
        axes = axes.reshape(1, -1)
    if num_samples_per_snr == 1:
        axes = axes.reshape(-1, 1)
    
    # 如果只有一个SNR和一个样本，axes是单个对象，需要转换为数组
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    
    for snr_idx, h5_file in enumerate(h5_files):
        # 获取SNR值
        snr_str = os.path.basename(h5_file).replace('snr_', '').replace('.h5', '')
        txt_file = os.path.join(h5_folder, f'snr_{snr_str}.txt')
        
        print(f'\n处理 SNR = {snr_str} dB')
        
        # 加载信号和元数据
        signals, num_meteors, signal_length, snr_dB = load_h5_signals(h5_file)
        metadata_df = load_metadata(txt_file)
        
        if metadata_df is None:
            print(f'  警告: 无法读取元数据文件 {txt_file}')
            # 使用默认值
            velocities = [None] * num_meteors
        else:
            # 提取速度信息（Vm列）
            velocities = []
            for idx in range(len(metadata_df)):
                row = metadata_df.iloc[idx]
                # 尝试不同的列名（CSV中列名是Vm）
                vm = None
                for col in ['Vm', 'vm', 'VM', 'VM ', 'Vm ']:
                    if col in row.index:
                        try:
                            vm = float(row[col])
                            break
                        except:
                            continue
                velocities.append(vm if vm is not None and not pd.isna(vm) else None)
        
        # 随机选择信号
        if num_meteors < num_samples_per_snr:
            selected_indices = list(range(num_meteors))
            print(f'  警告: 只有 {num_meteors} 个信号，少于请求的 {num_samples_per_snr} 个')
        else:
            selected_indices = np.random.choice(num_meteors, size=num_samples_per_snr, replace=False)
        
        # 绘制选中的信号
        for sample_idx, signal_idx in enumerate(selected_indices):
            ax = axes[snr_idx, sample_idx]
            signal = signals[signal_idx]
            
            # 时间轴（假设PRF=430Hz）
            PRF = 430.0
            time_axis = np.arange(len(signal)) / PRF
            
            # 绘制幅度
            ax.plot(time_axis, np.abs(signal), 'b-', linewidth=1.5, label='幅度')
            ax.set_xlabel('时间 (s)', fontsize=10)
            ax.set_ylabel('幅度', fontsize=10)
            
            # 获取速度信息
            velocity = velocities[signal_idx] if signal_idx < len(velocities) else None
            if velocity is not None and not pd.isna(velocity):
                title = f'SNR={snr_str} dB\nVm={velocity:.1f} m/s'
            else:
                title = f'SNR={snr_str} dB\nVm=N/A'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('随机选择的流星信号可视化', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图片
    output_file = os.path.join(h5_folder, 'random_signals_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\n图片已保存到: {output_file}')
    
    plt.show()

def visualize_phase_signals(h5_folder, num_samples_per_snr=3, random_seed=42):
    """
    可视化信号的相位（解包裹后）
    """
    np.random.seed(random_seed)
    
    h5_files = sorted(glob.glob(os.path.join(h5_folder, 'snr_*.h5')))
    
    if len(h5_files) == 0:
        print(f'在 {h5_folder} 中没有找到h5文件！')
        return
    
    num_snrs = len(h5_files)
    fig, axes = plt.subplots(num_snrs, num_samples_per_snr, 
                             figsize=(6*num_samples_per_snr, 4.5*num_snrs))
    
    if num_snrs == 1:
        axes = axes.reshape(1, -1)
    if num_samples_per_snr == 1:
        axes = axes.reshape(-1, 1)
    
    # 如果只有一个SNR和一个样本，axes是单个对象，需要转换为数组
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    
    for snr_idx, h5_file in enumerate(h5_files):
        snr_str = os.path.basename(h5_file).replace('snr_', '').replace('.h5', '')
        txt_file = os.path.join(h5_folder, f'snr_{snr_str}.txt')
        
        signals, num_meteors, signal_length, snr_dB = load_h5_signals(h5_file)
        metadata_df = load_metadata(txt_file)
        
        if metadata_df is not None:
            velocities = []
            for idx in range(len(metadata_df)):
                row = metadata_df.iloc[idx]
                vm = None
                for col in ['Vm', 'vm', 'VM']:
                    if col in row.index:
                        vm = row[col]
                        break
                velocities.append(vm if vm is not None and not pd.isna(vm) else None)
        else:
            velocities = [None] * num_meteors
        
        if num_meteors < num_samples_per_snr:
            selected_indices = list(range(num_meteors))
        else:
            selected_indices = np.random.choice(num_meteors, size=num_samples_per_snr, replace=False)
        
        for sample_idx, signal_idx in enumerate(selected_indices):
            ax = axes[snr_idx, sample_idx]
            signal = signals[signal_idx]
            
            PRF = 430.0
            time_axis = np.arange(len(signal)) / PRF
            
            # 解包裹相位
            phase = np.unwrap(np.angle(signal))
            
            ax.plot(time_axis, phase, 'r-', linewidth=1.5, label='相位')
            ax.set_xlabel('时间 (s)', fontsize=10)
            ax.set_ylabel('相位 (rad)', fontsize=10)
            
            velocity = velocities[signal_idx] if signal_idx < len(velocities) else None
            if velocity is not None and not pd.isna(velocity):
                title = f'SNR={snr_str} dB\nVm={velocity:.1f} m/s'
            else:
                title = f'SNR={snr_str} dB\nVm=N/A'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('随机选择的流星信号相位可视化', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = os.path.join(h5_folder, 'random_signals_phase_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\n相位图片已保存到: {output_file}')
    
    plt.show()

def estimate_velocity_from_phase(signal, PRF=430.0, lamda=2.99792458e8/35e6):
    """
    从信号相位估计速度（440点）
    返回: (Ve_estimated, fitA, fitB, p, A_time, As)
    """
    As = np.unwrap(np.angle(signal))
    P_dim = len(signal)
    A_time = np.arange(P_dim) / PRF
    B_phase = As[:min(len(As), P_dim//2)]
    
    min_len = min(len(A_time), len(B_phase))
    A_time = A_time[:min_len]
    B_phase = B_phase[:min_len]
    
    # 速度拟合方法
    if len(B_phase) > 50:
        start_ratio = 0.15
        end_ratio = 0.90
        start_idx = max(0, int(len(B_phase) * start_ratio))
        end_idx = min(int(len(B_phase) * end_ratio), len(B_phase))
        
        min_points = 40
        if (end_idx - start_idx) < min_points:
            center = len(B_phase) // 2
            half_range = min_points // 2
            start_idx = max(0, center - half_range)
            end_idx = min(len(B_phase), center + half_range)
        
        fit_range = np.arange(start_idx, end_idx)
        if len(fit_range) > 30:
            dB_range = np.diff(B_phase[fit_range])
            window_size = min(20, len(fit_range) // 3)
            if len(dB_range) > window_size * 2:
                kernel = np.ones(window_size) / window_size
                mean_rolling = np.convolve(dB_range, kernel, mode='valid')
                sq_mean_rolling = np.convolve(dB_range**2, kernel, mode='valid')
                std_dev_range = np.sqrt(sq_mean_rolling - mean_rolling**2)
                min_std_idx = np.argmin(std_dev_range)
                stable_start = fit_range[0] + min_std_idx
                stable_end = min(stable_start + int(len(fit_range) * 0.7), fit_range[-1] + 1)
                
                fitA = A_time[stable_start:stable_end]
                fitB = B_phase[stable_start:stable_end]
            else:
                fitA = A_time[start_idx:end_idx]
                fitB = B_phase[start_idx:end_idx]
        else:
            fitA = A_time[start_idx:end_idx]
            fitB = B_phase[start_idx:end_idx]
        
        if len(fitA) > 1:
            p = np.polyfit(fitA, fitB, 1)
            Ve_estimated = -lamda * p[0] / (4 * np.pi)
            
            if abs(Ve_estimated) > 10000 or not np.isfinite(Ve_estimated):
                start_idx = max(0, int(len(B_phase) * 0.2))
                end_idx = min(int(len(B_phase) * 0.8), len(B_phase))
                fitA = A_time[start_idx:end_idx]
                fitB = B_phase[start_idx:end_idx]
                p = np.polyfit(fitA, fitB, 1)
                Ve_estimated = -lamda * p[0] / (4 * np.pi)
        else:
            p = [0, 0]
            Ve_estimated = 0
    elif len(B_phase) > 10:
        start_idx = max(0, int(len(B_phase) * 0.2))
        end_idx = min(int(len(B_phase) * 0.8), len(B_phase))
        fitA = A_time[start_idx:end_idx]
        fitB = B_phase[start_idx:end_idx]
        p = np.polyfit(fitA, fitB, 1)
        Ve_estimated = -lamda * p[0] / (4 * np.pi)
    else:
        p = np.polyfit(A_time, B_phase, 1)
        Ve_estimated = -lamda * p[0] / (4 * np.pi)
        fitA = A_time
        fitB = B_phase
    
    return Ve_estimated, fitA, fitB, p, A_time, As

def coherent_integration(signal, n=4):
    """相干积累：每n个脉冲平均"""
    P_dim = len(signal)
    Esc = np.zeros(P_dim // n, dtype=complex)
    for i in range(P_dim // n):
        Esc[i] = np.mean(signal[i*n:(i+1)*n])
    return Esc

def estimate_velocity_coherent(signal, PRF=430.0, lamda=2.99792458e8/35e6, n=4):
    """
    从相干积累后的信号相位估计速度（110点）
    返回: (Ve_coherent, fitA_coherent, fitB_coherent, p_coherent, A_time_coherent, Asc, Esc)
    """
    Esc = coherent_integration(signal, n)
    Asc = np.unwrap(np.angle(Esc))
    P_dim = len(signal)
    A_time_coherent = np.arange(P_dim // n) * (n / PRF)
    B_phase_coherent = Asc[:min(len(Asc), (P_dim // n) // 2)]
    
    min_len = min(len(A_time_coherent), len(B_phase_coherent))
    A_time_coherent = A_time_coherent[:min_len]
    B_phase_coherent = B_phase_coherent[:min_len]
    
    # 速度拟合方法
    if len(B_phase_coherent) > 20:
        start_ratio = 0.15
        end_ratio = 0.90
        start_idx_coherent = max(0, int(len(B_phase_coherent) * start_ratio))
        end_idx_coherent = min(int(len(B_phase_coherent) * end_ratio), len(B_phase_coherent))
        
        min_points = 15
        if (end_idx_coherent - start_idx_coherent) < min_points:
            center = len(B_phase_coherent) // 2
            half_range = min_points // 2
            start_idx_coherent = max(0, center - half_range)
            end_idx_coherent = min(len(B_phase_coherent), center + half_range)
        
        fit_range_coherent = np.arange(start_idx_coherent, end_idx_coherent)
        if len(fit_range_coherent) > 10:
            dB_range_coherent = np.diff(B_phase_coherent[fit_range_coherent])
            window_size_coherent = min(10, len(fit_range_coherent) // 3)
            if len(dB_range_coherent) > window_size_coherent * 2:
                kernel_coh = np.ones(window_size_coherent) / window_size_coherent
                mean_rolling_coh = np.convolve(dB_range_coherent, kernel_coh, mode='valid')
                sq_mean_rolling_coh = np.convolve(dB_range_coherent**2, kernel_coh, mode='valid')
                std_dev_range_coherent = np.sqrt(sq_mean_rolling_coh - mean_rolling_coh**2)
                min_std_idx_coherent = np.argmin(std_dev_range_coherent)
                stable_start_coherent = fit_range_coherent[0] + min_std_idx_coherent
                stable_end_coherent = min(stable_start_coherent + int(len(fit_range_coherent) * 0.7), 
                                         fit_range_coherent[-1] + 1)
                
                fitA_coherent = A_time_coherent[stable_start_coherent:stable_end_coherent]
                fitB_coherent = B_phase_coherent[stable_start_coherent:stable_end_coherent]
            else:
                fitA_coherent = A_time_coherent[start_idx_coherent:end_idx_coherent]
                fitB_coherent = B_phase_coherent[start_idx_coherent:end_idx_coherent]
        else:
            fitA_coherent = A_time_coherent[start_idx_coherent:end_idx_coherent]
            fitB_coherent = B_phase_coherent[start_idx_coherent:end_idx_coherent]
        
        if len(fitA_coherent) > 1:
            p_coherent = np.polyfit(fitA_coherent, fitB_coherent, 1)
            Ve_coherent = -lamda * p_coherent[0] / (4 * np.pi)
            
            if abs(Ve_coherent) > 10000 or not np.isfinite(Ve_coherent):
                start_idx_coherent = max(0, int(len(B_phase_coherent) * 0.2))
                end_idx_coherent = min(int(len(B_phase_coherent) * 0.8), len(B_phase_coherent))
                fitA_coherent = A_time_coherent[start_idx_coherent:end_idx_coherent]
                fitB_coherent = B_phase_coherent[start_idx_coherent:end_idx_coherent]
                p_coherent = np.polyfit(fitA_coherent, fitB_coherent, 1)
                Ve_coherent = -lamda * p_coherent[0] / (4 * np.pi)
        else:
            p_coherent = [0, 0]
            Ve_coherent = 0
    elif len(B_phase_coherent) > 5:
        start_idx_coherent = max(0, int(len(B_phase_coherent) * 0.2))
        end_idx_coherent = min(int(len(B_phase_coherent) * 0.8), len(B_phase_coherent))
        fitA_coherent = A_time_coherent[start_idx_coherent:end_idx_coherent]
        fitB_coherent = B_phase_coherent[start_idx_coherent:end_idx_coherent]
        p_coherent = np.polyfit(fitA_coherent, fitB_coherent, 1)
        Ve_coherent = -lamda * p_coherent[0] / (4 * np.pi)
    else:
        p_coherent = np.polyfit(A_time_coherent, B_phase_coherent, 1)
        Ve_coherent = -lamda * p_coherent[0] / (4 * np.pi)
        fitA_coherent = A_time_coherent
        fitB_coherent = B_phase_coherent
    
    return Ve_coherent, fitA_coherent, fitB_coherent, p_coherent, A_time_coherent, Asc, Esc

def visualize_meteors_multi_snr(h5_folder, num_meteors=3, random_seed=42):
    """
    选择3个流星，展示它们在不同SNR下的信号
    每个流星4个子图：440点幅度、440点相位+拟合、110点幅度、110点相位+拟合
    """
    np.random.seed(random_seed)
    
    # 查找所有h5文件
    h5_files = sorted(glob.glob(os.path.join(h5_folder, 'snr_*.h5')))
    
    if len(h5_files) == 0:
        print(f'在 {h5_folder} 中没有找到h5文件！')
        return
    
    print(f'找到 {len(h5_files)} 个h5文件')
    
    # 从第一个文件读取信号数量，选择流星索引
    signals_first, num_meteors_total, _, _ = load_h5_signals(h5_files[0])
    if num_meteors_total < num_meteors:
        selected_indices = list(range(num_meteors_total))
        print(f'  警告: 只有 {num_meteors_total} 个信号，少于请求的 {num_meteors} 个')
    else:
        selected_indices = np.random.choice(num_meteors_total, size=num_meteors, replace=False)
    
    print(f'选择的流星索引: {selected_indices}')
    
    # 读取所有SNR的信号和元数据
    all_signals = {}  # {snr_str: signals_array}
    all_metadata = {}  # {snr_str: metadata_df}
    all_vr = {}  # {snr_str: [vr_list]}
    
    for h5_file in h5_files:
        snr_str = os.path.basename(h5_file).replace('snr_', '').replace('.h5', '')
        txt_file = os.path.join(h5_folder, f'snr_{snr_str}.txt')
        
        signals, _, _, _ = load_h5_signals(h5_file)
        all_signals[snr_str] = signals
        
        metadata_df = load_metadata(txt_file)
        all_metadata[snr_str] = metadata_df
        
        # 提取Vr（径向速度）
        if metadata_df is not None:
            vr_list = []
            for idx in range(len(metadata_df)):
                row = metadata_df.iloc[idx]
                vr = None
                for col in ['Vr', 'vr', 'VR', 'VR ', 'Vr ']:
                    if col in row.index:
                        try:
                            vr = float(row[col])
                            break
                        except:
                            continue
                vr_list.append(vr if vr is not None and not pd.isna(vr) else None)
            all_vr[snr_str] = vr_list
        else:
            all_vr[snr_str] = [None] * len(signals)
    
    # 物理常数
    PRF = 430.0
    c = 2.99792458e8
    f = 35e6
    lamda = c / f
    
    # 创建图形：3个流星，每个4个子图
    fig, axes = plt.subplots(num_meteors, 4, figsize=(20, 5*num_meteors))
    if num_meteors == 1:
        axes = axes.reshape(1, -1)
    
    snr_strs = sorted(all_signals.keys(), key=lambda x: float('inf') if x == 'inf' else float(x))
    colors = plt.cm.tab10(np.linspace(0, 1, len(snr_strs)))
    
    for meteor_idx, signal_idx in enumerate(selected_indices):
        # 获取Vr（从第一个SNR的元数据）
        first_snr = snr_strs[0]
        vr_value = None
        if signal_idx < len(all_vr[first_snr]):
            vr_value = all_vr[first_snr][signal_idx]
        
        # 子图1: 440点幅度
        ax1 = axes[meteor_idx, 0]
        for snr_idx, snr_str in enumerate(snr_strs):
            signal = all_signals[snr_str][signal_idx]
            time_axis = np.arange(len(signal)) / PRF
            ax1.plot(time_axis, np.abs(signal), '-', linewidth=1.5, 
                    color=colors[snr_idx], label=f'SNR={snr_str} dB', alpha=0.7)
        ax1.set_xlabel('时间 (s)', fontsize=10)
        ax1.set_ylabel('幅度', fontsize=10)
        ax1.set_title(f'流星 {meteor_idx+1}: 440点幅度' + (f' (Vr={vr_value:.1f} m/s)' if vr_value is not None else ''), 
                     fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)
        
        # 子图2: 440点相位+拟合
        ax2 = axes[meteor_idx, 1]
        for snr_idx, snr_str in enumerate(snr_strs):
            signal = all_signals[snr_str][signal_idx]
            Ve_est, fitA, fitB, p, A_time, As = estimate_velocity_from_phase(signal, PRF, lamda)
            
            ax2.plot(A_time, As[:len(A_time)], '-', linewidth=1.5, 
                    color=colors[snr_idx], label=f'SNR={snr_str} dB (Vr={Ve_est:.1f})', alpha=0.7)
            if len(fitA) > 0:
                ax2.plot(fitA, np.polyval(p, fitA), '--', linewidth=2, 
                        color=colors[snr_idx], alpha=0.8)
        ax2.set_xlabel('时间 (s)', fontsize=10)
        ax2.set_ylabel('相位 (rad)', fontsize=10)
        ax2.set_title(f'流星 {meteor_idx+1}: 440点相位+拟合', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)
        
        # 子图3: 110点幅度（相干积累）
        ax3 = axes[meteor_idx, 2]
        for snr_idx, snr_str in enumerate(snr_strs):
            signal = all_signals[snr_str][signal_idx]
            Esc = coherent_integration(signal, n=4)
            time_axis_coh = np.arange(len(Esc)) * (4 / PRF)
            ax3.plot(time_axis_coh, np.abs(Esc), '-', linewidth=1.5, 
                    color=colors[snr_idx], label=f'SNR={snr_str} dB', alpha=0.7)
        ax3.set_xlabel('时间 (s)', fontsize=10)
        ax3.set_ylabel('幅度', fontsize=10)
        ax3.set_title(f'流星 {meteor_idx+1}: 110点幅度（相干积累）', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=8)
        
        # 子图4: 110点相位+拟合
        ax4 = axes[meteor_idx, 3]
        for snr_idx, snr_str in enumerate(snr_strs):
            signal = all_signals[snr_str][signal_idx]
            Ve_coh, fitA_coh, fitB_coh, p_coh, A_time_coh, Asc, Esc = estimate_velocity_coherent(signal, PRF, lamda, n=4)
            
            ax4.plot(A_time_coh, Asc[:len(A_time_coh)], '-', linewidth=1.5, 
                    color=colors[snr_idx], label=f'SNR={snr_str} dB (Vr={Ve_coh:.1f})', alpha=0.7)
            if len(fitA_coh) > 0:
                ax4.plot(fitA_coh, np.polyval(p_coh, fitA_coh), '--', linewidth=2, 
                        color=colors[snr_idx], alpha=0.8)
        ax4.set_xlabel('时间 (s)', fontsize=10)
        ax4.set_ylabel('相位 (rad)', fontsize=10)
        ax4.set_title(f'流星 {meteor_idx+1}: 110点相位+拟合', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=8)
    
    plt.suptitle('3个流星在不同SNR下的信号分析', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图片
    output_file = os.path.join(h5_folder, 'meteors_multi_snr_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\n图片已保存到: {output_file}')
    
    plt.show()

if __name__ == "__main__":
    # 配置路径
    h5_folder = r'F:\denoisetest'
    
    print('=' * 60)
    print('流星信号可视化')
    print('=' * 60)
    print(f'H5文件文件夹: {h5_folder}')
    print('=' * 60)
    
    # 新的可视化：3个流星，不同SNR
    print('\n生成多SNR对比可视化...')
    visualize_meteors_multi_snr(h5_folder, num_meteors=3, random_seed=42)
    
    print('\n' + '=' * 60)
    print('可视化完成！')
    print('=' * 60)

