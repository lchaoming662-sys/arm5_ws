#!/usr/bin/env python3
"""_diag_normal3.py — k 与阈值的一次性联合标定（最终诊断）

前两轮已经排除的两条路：
  · 指标形式「夹角 σ」：混进坏数据反而变小（第2 轮实测 22.19 → 21.x），已废弃
  · 单纯调阈值：干净工况就已被判不合格（第 1 轮 22° vs 15°），必须先改 k

本轮要一次定下两件事：
  1. k 近邻数取多少 —— 决定法向估计的噪声水平
  2. 阈值取多少 —— 决定「什么程度的污染被拒」

设计成一个**决策表**而不是单点测量：横轴是污染比例，纵轴是 k，
每格是 1-|cos| 均值。我们要的是「干净列全部低于阈值、污染列全部高于」
的那个 k 与阈值。判据因此有可验证的分隔度，而不是一个孤立数字。
"""
import os
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry_validity as gv  # noqa: E402

S = 0.025
DENSITY = 0.00046        # 实测网格间距 0.46 mm（见 _diag_normal.py 的反算）


def plate(noise=0.0013, tilt=0.21, seed=0):
    rng = np.random.default_rng(seed)
    n = max(int(round(S / DENSITY)), 4)
    gx = np.linspace(-S / 2, S / 2, n)
    gy = np.linspace(-S / 2, S / 2, n)
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    X, Y = X.ravel(), Y.ravel()
    p = np.stack([X, Y, tilt * X], axis=1)
    p = p + rng.normal(0, 0.25 * DENSITY, p.shape)
    p[:, 2] += rng.normal(0, noise, len(p))
    return p


def rot(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def score(p, k):
    nrm, _ = gv.point_normals(p, k=k, use_open3d=False)
    q = p - p.mean(axis=0)
    _, vecs = np.linalg.eigh((q.T @ q) / (len(q) - 1))
    mn = vecs[:, 0]
    unit = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return float((1.0 - np.abs(unit @ mn)).mean())


def main():
    flat = plate(seed=5)
    wall = plate(seed=6) @ rot(90).T + np.array([0.0, 0.0135, 0.0])

    ks = [20, 30, 40, 50, 64, 80]
    fracs = [0.0, 0.05, 0.10, 0.20, 0.35]
    print('=== 1-|cos| 均值：行 = 污染比例，列 = k 近邻数 ===')
    print('干净列必须低、污染列必须高；且要留出明显的分隔。\n')
    print(f'{"污染%":>7s}' + ''.join(f'{"k=" + str(k):>9s}' for k in ks))
    print('-' * (7 + 9 * len(ks)))
    table = {}
    for fr in fracs:
        if fr == 0:
            pts = flat
        else:
            w = int(round(fr / (1 - fr) * len(flat)))
            pts = np.vstack([flat, wall[:w]])
        row = []
        for k in ks:
            v = score(pts, k)
            table[(fr, k)] = v
            row.append(f'{v:9.3f}')
        print(f'{fr*100:6.0f}%' + ''.join(row))

    print('\n=== 分隔度：污染 10% 减去干净，值越大越好判 ===')
    print(f'{"k":>5s}{"干净":>10s}{"污染10%":>10s}{"间隔":>10s}')
    print('-' * 35)
    best = None
    for k in ks:
        a, b = table[(0.0, k)], table[(0.10, k)]
        gap = b - a
        mid = 0.5 * (a + b)
        print(f'{k:5d}{a:10.3f}{b:10.3f}{gap:10.3f}   建议阈值 ≈ {mid:.3f}')
        # 同时要求：干净值明显低于中点、污染值明显高于中点
        if best is None or (gap, -mid) > (best[1], -best[2]):
            best = (k, gap, mid)

    k, gap, mid = best
    print(f'\n选定：k = {k}，阈值 = {mid:.3f}（取干净与污染10% 的中点）')
    print(f'此时干净工况余量 {mid - table[(0.0, k)]:.3f}，'
          f'污染 5% 余量 {table[(0.05, k)] - mid:+.3f}')
    print('\n注意这个阈值只对「平面内法向偏离」敏感，对「整体倾斜」不敏感')
    print('（倾斜由平面性 λ₃/λ₁ 管），两者分工不同，不要互相替代。')
    print(f'\n各k 下污染 5% 的取值：' +
          ' '.join(f'k={k}:{table[(0.05, k)]:.3f}' for k in ks))


if __name__ == '__main__':
    main()