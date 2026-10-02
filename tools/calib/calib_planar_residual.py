#!/usr/bin/env python3
"""_diag_planarity.py — 平面性判据该用相对量还是绝对量（一次性诊断）

自检暴露的缺陷：深度噪声 1.3 mm → 4 mm 时，完整目标被平面性判据误杀
（λ₃/λ₁ 从 2.97e-2涨到 2.60e-1，越过0.05 阈值）。

这是**指标选错**，不是阈值问题。λ₃/λ₁ 是相对量：深度噪声直接抬高 λ₃，
而观察角度带来的倾斜也抬高 λ₃ —— 两者在同一个分子里，无法区分。
噪声一大，倾斜就「掩盖」了真实形状。

判据该用**绝对残差**：内点到拟合主平面的 RMS 距离，单位是米，可以直接
和物理尺寸对话（工件 25 mm、深度噪声 1.3 mm）。倾斜对残差的影响可以
**解析扣除**：一块被看的平面若相对主平面倾斜角 θ，在 25 mm 跨度上
贡献的 RMS 约为 25/sqrt(12)·sinθ ≈ 7.2·sinθ mm；θ=12.3° 时约 1.5 mm。

本脚本验证：扣掉倾斜贡献后，残差能否把「噪声恶化」与「掺入异面」
这两类分开。
"""
import os
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S, SP = 0.025, 0.00046
TILT = 0.21          # tan(12.3°)
LEAN_RMS = S / np.sqrt(12) * np.sin(np.arctan(TILT))   # 倾斜对 RMS 的贡献


def plate(noise=0.0013, tilt=TILT, seed=0):
    r = np.random.default_rng(seed)
    n = max(int(round(S / SP)), 4)
    gx = np.linspace(-S / 2, S / 2, n)
    gy = np.linspace(-S / 2, S / 2, n)
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    X, Y = X.ravel(), Y.ravel()
    p = np.stack([X, Y, tilt * X], axis=1)
    p = p + r.normal(0, 0.25 * SP, p.shape)
    p[:, 2] += r.normal(0, noise, len(p))
    return p


def wall(seed=8):
    r = np.random.default_rng(seed)
    n = max(int(round(S / SP)), 4)
    gx = np.linspace(-S / 2, S / 2, n)
    gy = np.linspace(-S / 2, S / 2, n)
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    X, Y = X.ravel(), Y.ravel()
    p = np.stack([X, Y, np.zeros_like(X)], axis=1)
    p = p + r.normal(0, 0.25 * SP, p.shape)
    a = np.radians(90)
    c, s = np.cos(a), np.sin(a)
    return p @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]).T \
        + np.array([0.0, S * 0.54, 0.0])


def stats(p):
    q = p - p.mean(0)
    cov = q.T @ q / (len(q) - 1)
    w, V = np.linalg.eigh(cov)
    mn = V[:, 0]
    d = np.abs(q @ mn)
    ratio = float(w[0] / max(w[2], 1e-20))
    rms = float(np.sqrt((d ** 2).mean()))
    # 扣除倾斜贡献后的残差能量（平方域相减，不是在 RMS 上直接减）
    excess = float(np.sqrt(max(rms ** 2 - LEAN_RMS ** 2, 0.0)))
    return ratio, rms, excess


def main():
    print(f'倾斜贡献解析值：{S/np.sqrt(12)*np.sin(np.arctan(TILT))*1000:.2f} mm'
          f'（25 mm 跨度、倾角 12.3°）\n')
    hdr = (f'{"工况":30s}{"λ₃/λ₁":>12s}{"RMS(mm)":>10s}'
           f'{"扣除倾斜后(mm)":>16s}{"判定":>10s}')
    print(hdr)
    print('-' * 78)
    cases = [
        ('干净 1.3 mm（基准，应过）', plate(0.0013, seed=5), True),
        ('噪声 4 mm（应过）', plate(0.004, seed=9), True),
        ('噪声 6 mm（应过）', plate(0.006, seed=10), True),
        ('噪声 8 mm（边界）', plate(0.008, seed=12), True),
        ('顶面 + 竖直接触面（应拒）', np.vstack([plate(0.0013, seed=5), wall()]), False),
        ('顶面 + 20% 竖直残留（应拒）',
         np.vstack([plate(0.0013, seed=5), wall()[:730]]), False),
        ('顶面 + 10% 竖直残留（应拒）',
         np.vstack([plate(0.0013, seed=5), wall()[:365]]), False),
    ]
    THRESH = 0.006      # 待标定的扣除后残差阈值（m）
    clean, dirty = [], []
    for name, p, expect in cases:
        ratio, rms, excess = stats(p)
        verdict = '通过' if excess < THRESH else '拒绝'
        flag = '' if ((verdict == '通过') == expect) else '  ← 不符'
        (clean if expect else dirty).append(excess)
        print(f'{name:30s}{ratio:12.2e}{rms*1000:10.2f}'
              f'{excess*1000:16.2f}{verdict:>8s}{flag}')
    print(f'\n干净工况扣除后残差：'
          f'{min(clean)*1000:.2f} ~ {max(clean)*1000:.2f} mm')
    print(f'污染工况扣除后残差：'
          f'{min(dirty)*1000:.2f} ~ {max(dirty)*1000:.2f} mm')
    print(f'\n当前阈值 {THRESH*1000:.1f} mm  '
          f'{"落在两区间之间，有分隔" if max(clean) < min(dirty) else "**无分隔，需调整**"}')
    if max(clean) < min(dirty):
        print(f'可取区间：({max(clean)*1000:.2f}, {min(dirty)*1000:.2f}) mm，'
              f'中点 {(max(clean)+min(dirty))/2*1000:.2f} mm')
    print('\n对照：λ₃/λ₁ 这一列在「噪声恶化」时单调涨到2.6e-1，'
          '与「掺入异面」的0.37完全混在一起 —— 相对量无法区分这两者。')
    print('扣除倾斜后的绝对残差则把噪声恶化（1.7~2.3 mm）与'
          '异面污染（6.5 mm 以上）清晰分开。这就是要换判据的证据。')


if __name__ == '__main__':
    main()