#!/usr/bin/env python3
"""
光学衍射仿真 — 3D 交互式界面
菲涅耳衍射 & 夫琅禾费衍射 (单缝/圆孔/圆屏/矩孔)
Python Flask 后端 + Three.js 3D 前端
"""

import json
import math
import io
import base64
import numpy as np
from scipy.special import fresnel, j1
from flask import Flask, request, jsonify, send_from_directory
import threading
import webbrowser
import os

app = Flask(__name__, static_folder='static')
os.makedirs('static', exist_ok=True)

# ============================================================
# 物理计算函数
# ============================================================

def wavelength_to_rgb(wl_nm):
    """波长 (nm) 转 RGB"""
    if wl_nm < 380:
        return (0.5, 0.0, 0.5)
    elif wl_nm < 440:
        r = -(wl_nm - 440) / (440 - 380)
        g, b = 0.0, 1.0
    elif wl_nm < 490:
        r, g = 0.0, (wl_nm - 440) / (490 - 440)
        b = 1.0
    elif wl_nm < 510:
        r, g = 0.0, 1.0
        b = -(wl_nm - 510) / (510 - 490)
    elif wl_nm < 580:
        r = (wl_nm - 510) / (580 - 510)
        g, b = 1.0, 0.0
    elif wl_nm < 645:
        r, g = 1.0, -(wl_nm - 645) / (645 - 580)
        b = 0.0
    elif wl_nm <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    # Gamma 校正
    def gamma(c):
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1/2.4)) - 0.055
    return (gamma(r), gamma(g), gamma(b))


def compute_fresnel_number(diff_type, lam, a_or_radius, L):
    """计算菲涅耳数 Nf"""
    if lam * L == 0:
        return 0
    return a_or_radius ** 2 / (lam * L)


def compute_intensity_1d(diff_type, diff_mode, lam, a, b, disk_radius, z, screen_half_mm=20):
    """计算1D光强分布"""
    x_line = np.linspace(-screen_half_mm, screen_half_mm, 300)

    if diff_type == "single_slit":
        a_m = a * 1e-3
        if diff_mode == "fresnel":
            u = x_line * 1e-3 * np.sqrt(2 / (lam * z))
            S_u, C_u = fresnel(u)
            I = C_u**2 + S_u**2
        else:
            theta = np.arctan(x_line * 1e-3 / z)
            beta = np.pi * a_m * np.sin(theta) / lam
            beta = np.where(np.abs(beta) < 1e-10, 1e-10, beta)
            I = (np.sin(beta) / beta) ** 2

    elif diff_type == "rectangular":
        a_m = a * 1e-3
        b_m = b * 1e-3
        theta = np.arctan(x_line * 1e-3 / z)
        alpha = np.pi * a_m * np.sin(theta) / lam
        alpha = np.where(np.abs(alpha) < 1e-10, 1e-10, alpha)
        I_x = (np.sin(alpha) / alpha) ** 2
        beta = np.pi * b_m * np.sin(theta) / lam
        beta = np.where(np.abs(beta) < 1e-10, 1e-10, beta)
        I_y = (np.sin(beta) / beta) ** 2
        # Return both
        I_x = I_x / np.max(I_x) if np.max(I_x) > 0 else I_x
        I_y = I_y / np.max(I_y) if np.max(I_y) > 0 else I_y
        return {
            'x': x_line.tolist(),
            'I_x': I_x.tolist(),
            'I_y': I_y.tolist(),
            'Nf': compute_fresnel_number(diff_type, lam, a_m, z),
            'mode': diff_mode
        }

    elif diff_type == "circular_disk":
        a_m = disk_radius * 1e-3
        if diff_mode == "fresnel":
            rho_a = a_m * np.sqrt(2 / (lam * z))
            rho = np.abs(x_line) * 1e-3 * np.sqrt(2 / (lam * z))
            C_a, S_a = fresnel(rho_a)
            I = np.zeros_like(rho)
            I_center = 0.5 * ((0.5 - C_a)**2 + (0.5 - S_a)**2)
            mask_in = np.abs(x_line) * 1e-3 <= a_m
            I[mask_in] = I_center
            mask_out = ~mask_in
            if np.any(mask_out):
                C_out, S_out = fresnel(rho[mask_out])
                C_diff = C_out - C_a
                S_diff = S_out - S_a
                I[mask_out] = 0.5 * ((0.5 - C_diff)**2 + (0.5 - S_diff)**2)
        else:
            # Fraunhofer 圆屏衍射 — 巴比涅原理: 圆屏衍射 = 自由传播 - 圆孔衍射
            D = 2 * a_m
            theta = np.arctan(np.abs(x_line) * 1e-3 / z)
            u = np.pi * D * np.sin(theta) / lam
            u = np.where(np.abs(u) < 1e-10, 1e-10, u)
            I_hole = (2 * j1(u) / u) ** 2
            I = np.abs(1.0 - I_hole) ** 2

    elif diff_type == "circular":
        D = a * 1e-3  # diameter
        if diff_mode == "fresnel":
            rho = x_line * 1e-3 * np.sqrt(2 / (lam * z))
            rho_max = (D / 2) * np.sqrt(2 / (lam * z))
            I = np.zeros_like(rho)
            mask = np.abs(rho) <= rho_max
            inner = mask & (np.abs(rho) > 0.001)
            I[inner] = (2 * j1(rho[inner]) / rho[inner])**2
            if np.max(I[mask]) > 0:
                I[mask] = I[mask] / np.max(I[mask])
        else:
            theta = np.arctan(x_line * 1e-3 / z)
            u = np.pi * D * np.sin(theta) / lam
            u = np.where(np.abs(u) < 1e-10, 1e-10, u)
            I = (2 * j1(u) / u) ** 2

    # Normalize
    I = I / np.max(I) if np.max(I) > 0 else I

    a_m = a * 1e-3 if diff_type != "circular_disk" else disk_radius * 1e-3
    return {
        'x': x_line.tolist(),
        'I': I.tolist(),
        'Nf': compute_fresnel_number(diff_type, lam, 
                                     a_m if diff_type != "circular" else D, z),
        'mode': diff_mode
    }


def compute_pattern_2d(diff_type, diff_mode, lam, a, b, disk_radius, z, 
                         screen_half_mm=15, n_pixels=150, wavelength_nm=632.8):
    """计算2D衍射图样"""
    x_s = np.linspace(-screen_half_mm, screen_half_mm, n_pixels)
    y_s = np.linspace(-screen_half_mm, screen_half_mm, n_pixels)
    X_s, Y_s = np.meshgrid(x_s, y_s)
    R_s = np.sqrt(X_s**2 + Y_s**2)

    if diff_type == "single_slit":
        if diff_mode == "fresnel":
            u = X_s * 1e-3 * np.sqrt(2 / (lam * z))
            S_u, C_u = fresnel(u)
            I = C_u**2 + S_u**2
        else:
            a_m = a * 1e-3
            theta = np.arctan(X_s * 1e-3 / z)
            beta = np.pi * a_m * np.sin(theta) / lam
            beta = np.where(np.abs(beta) < 1e-10, 1e-10, beta)
            I = (np.sin(beta) / beta) ** 2

    elif diff_type == "rectangular":
        a_m = a * 1e-3
        b_m = b * 1e-3
        theta_x = np.arctan(X_s * 1e-3 / z)
        theta_y = np.arctan(Y_s * 1e-3 / z)
        alpha = np.pi * a_m * np.sin(theta_x) / lam
        beta = np.pi * b_m * np.sin(theta_y) / lam
        alpha = np.where(np.abs(alpha) < 1e-10, 1e-10, alpha)
        beta = np.where(np.abs(beta) < 1e-10, 1e-10, beta)
        I = (np.sin(alpha) / alpha) ** 2 * (np.sin(beta) / beta) ** 2

    elif diff_type == "circular":
        D = a * 1e-3
        if diff_mode == "fresnel":
            rho = R_s * 1e-3 * np.sqrt(2 / (lam * z))
            I = np.zeros_like(R_s)
            mask = rho > 0.001
            I[mask] = (2 * j1(rho[mask]) / rho[mask])**2
        else:
            theta = np.arctan(R_s * 1e-3 / z)
            u = np.pi * D * np.sin(theta) / lam
            u = np.where(np.abs(u) < 1e-10, 1e-10, u)
            I = (2 * j1(u) / u) ** 2

    elif diff_type == "circular_disk":
        a_m = disk_radius * 1e-3
        if diff_mode == "fresnel":
            # Fresnel circular disk - Poisson spot
            rho_a = a_m * np.sqrt(2 / (lam * z))
            rho = R_s * 1e-3 * np.sqrt(2 / (lam * z))
            C_a, S_a = fresnel(rho_a)
            I_center = 0.5 * ((0.5 - C_a)**2 + (0.5 - S_a)**2)
            I = np.zeros_like(R_s)
            mask_in = (R_s * 1e-3) <= a_m
            I[mask_in] = I_center
            mask_out = ~mask_in
            if np.any(mask_out):
                C_out, S_out = fresnel(rho[mask_out])
                C_diff = C_out - C_a
                S_diff = S_out - S_a
                I[mask_out] = 0.5 * ((0.5 - C_diff)**2 + (0.5 - S_diff)**2)
        else:
            # Fraunhofer 圆屏衍射 — 巴比涅原理: 圆屏衍射 = 自由传播 - 圆孔衍射
            D = 2 * disk_radius * 1e-3
            theta = np.arctan(R_s * 1e-3 / z)
            u = np.pi * D * np.sin(theta) / lam
            u = np.where(np.abs(u) < 1e-10, 1e-10, u)
            I_hole = (2 * j1(u) / u) ** 2
            I = np.abs(1.0 - I_hole) ** 2

    I = I / np.max(I) if np.max(I) > 0 else I

    # 幂律增强：提升弱信号（外圈衍射环）的可见度
    I_disp = np.power(np.clip(I, 0, 1), 0.4)

    # 黑底 + 波长色调 - 根据入射光波长自动配色
    wl_r, wl_g, wl_b = wavelength_to_rgb(wavelength_nm)
    brightness = np.clip(I_disp * 1.5, 0, 1)
    
    rgb = np.zeros((n_pixels, n_pixels, 3))
    rgb[:,:,0] = brightness * wl_r
    rgb[:,:,1] = brightness * wl_g
    rgb[:,:,2] = brightness * wl_b

    rgb = np.clip(rgb, 0, 1)

    # Convert to base64 PNG
    from PIL import Image
    img = (rgb * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, 'RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def get_wavefront_color(lam):
    """Get wavelength color as hex string"""
    rgb = wavelength_to_rgb(lam * 1e9)
    r = int(np.clip(rgb[0] * 255, 0, 255))
    g = int(np.clip(rgb[1] * 255, 0, 255))
    b = int(np.clip(rgb[2] * 255, 0, 255))
    return f'#{r:02x}{g:02x}{b:02x}'


# ============================================================
# API 路由
# ============================================================

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/calculate', methods=['POST'])
def calculate():
    """主计算 API"""
    data = request.json
    diff_type = data.get('diff_type', 'single_slit')
    diff_mode = data.get('diff_mode', 'fresnel')
    wavelength = data.get('wavelength', 632.8)  # nm
    aperture = data.get('aperture', 0.15)       # mm
    aperture_height = data.get('aperture_height', 0.5)  # mm
    disk_radius = data.get('disk_radius', 0.5)  # mm
    obs_distance = data.get('obs_distance', 50) # cm
    source_distance = data.get('source_distance', 30)  # cm (Fresnel only)

    lam = wavelength * 1e-9  # m
    z = obs_distance * 1e-2  # m

    a_or_radius = aperture * 1e-3  # m
    if diff_type == "circular_disk":
        a_or_radius = disk_radius * 1e-3

    # 菲涅耳衍射使用有效距离: 1/z_eff = 1/u + 1/v
    if diff_mode == "fresnel":
        u = source_distance * 1e-2  # 光源到孔径 (m)
        z_eff = u * z / (u + z)  # 有效传播距离
    else:
        z_eff = z
        u = z  # not used in Fraunhofer

    Nf = compute_fresnel_number(diff_type, lam, a_or_radius, z_eff)

    # Get 1D intensity distribution
    result_1d = compute_intensity_1d(diff_type, diff_mode, lam, 
                                     aperture, aperture_height, disk_radius, z_eff if diff_mode == "fresnel" else z)

    # 2D pattern 所用的屏幕范围: 按衍射特征尺寸自适应
    # 因子设为8~10，确保显示足够多的衍射环/条纹（参考初版界面要求）
    if diff_type == "single_slit":
        first_min_mm = z_eff * 1e3 * lam / (aperture * 1e-3)
        screen_half = max(first_min_mm * 10, 8)
    elif diff_type == "circular" or diff_type == "circular_disk":
        D = (aperture if diff_type == "circular" else disk_radius * 2) * 1e-3
        airy_mm = 1.22 * lam * z_eff / D * 1e3 if D > 0 else 10
        screen_half = max(airy_mm * 8, 8)
    elif diff_type == "rectangular":
        a_m = aperture * 1e-3
        b_m = aperture_height * 1e-3
        min_x = z_eff * 1e3 * lam / a_m
        min_y = z_eff * 1e3 * lam / b_m
        screen_half = max(min(min_x, min_y) * 8, 12)
    else:
        screen_half = 20

    # Get 2D pattern as base64 image - 高分辨率1024
    try:
        pattern_b64 = compute_pattern_2d(diff_type, diff_mode, lam, 
                                          aperture, aperture_height, disk_radius, z_eff if diff_mode == "fresnel" else z,
                                          n_pixels=1024, screen_half_mm=screen_half, wavelength_nm=wavelength)
    except ImportError:
        pattern_b64 = None

    # Color
    wl_color = get_wavefront_color(lam)

    # Compute first minimum/feature positions
    features = {}
    a_m = aperture * 1e-3
    if diff_type in ("single_slit", "circular"):
        if diff_mode == "fraunhofer":
            if diff_type == "single_slit":
                first_min = z * np.arcsin(lam / a_m) * 1e3 if lam/a_m <= 1 else z * 1e3
                features['first_min_mm'] = first_min
            else:
                airy_radius = 1.22 * lam * z / a_m * 1e3
                features['airy_radius_mm'] = airy_radius
    elif diff_type == "rectangular":
        b_m = aperture_height * 1e-3
        min_x = z * np.arcsin(lam / a_m) * 1e3 if lam/a_m <= 1 else z * 1e3
        min_y = z * np.arcsin(lam / b_m) * 1e3 if lam/b_m <= 1 else z * 1e3
        features['first_min_x'] = min_x
        features['first_min_y'] = min_y
    elif diff_type == "circular_disk":
        a_m_d = disk_radius * 1e-3
        if diff_mode == "fraunhofer":
            features['poisson_angle_deg'] = np.arcsin(lam / a_m_d) * 180/np.pi if lam/a_m_d <= 1 else None

    return jsonify({
        'Nf': Nf,
        'intensity': result_1d,
        'pattern_b64': pattern_b64,
        'color': wl_color,
        'features': features,
        'mode_suggestion': 'fraunhofer' if Nf < 1 else 'fresnel'
    })


@app.route('/api/knowledge', methods=['GET'])
def get_knowledge():
    """返回知识点内容"""
    return jsonify({
        'fresnel': {
            'title': '菲涅耳衍射（近场衍射）',
            'condition': 'Nf = a²/(λL) > 1',
            'features': [
                '衍射图样随观察距离变化',
                '球面波入射或出射',
                '图样形状与障碍物形状有关',
                '使用菲涅耳积分计算'
            ],
            'formula': 'I = C² + S² (菲涅耳积分)'
        },
        'fraunhofer': {
            'title': '夫琅禾费衍射（远场衍射）',
            'condition': 'Nf = a²/(λL) << 1',
            'features': [
                '观察距离须远大于衍射孔的Fresnel特征长度',
                '入射光为平行光（需透镜）',
                '衍射图样与观察距离无关（角度分布）',
                '满足夫琅禾费远场条件'
            ],
            'single_slit': 'I = I₀(sinβ/β)², β = πa·sinθ/λ',
            'single_slit_pattern': '中央亮纹最宽最亮，两侧亮纹逐渐变窄变暗，亮纹间距不等。波长越长或缝宽越小，中央条纹越宽越暗。',
            'single_slit_condition': '明纹条件: asinθ = (2k+1)λ/2 (k=1,2,3...)\n暗纹条件: asinθ = kλ (k=1,2,3...)',
            'circular': 'I = I₀[2J₁(u)/u]², u = πD·sinθ/λ, 艾里斑角半径 = 1.22λ/D',
            'circular_pattern': '中央为明亮的艾里斑，周围为明暗相间的同心圆环，中央亮斑集中了约84%的能量。',
            'circular_condition': '暗纹条件: D·sinθ = kλ (k=1,2,3...)，第一暗纹对应艾里斑边界。',
            'rectangular': 'I = I₀(sinα/α)²(sinβ/β)², α = πa·sinθx/λ, β = πb·sinθy/λ',
            'rectangular_pattern': '沿x和y方向分别衍射，在两个方向上产生不等间距的亮暗条纹。',
            'rectangular_condition': '暗纹条件: asinθx = k₁λ 且 bsinθy = k₂λ',
            'circular_disk': '巴比涅原理: I_disk = I_0 - I_hole',
            'circular_disk_pattern': '夫琅禾费圆屏衍射中心为暗斑；菲涅耳圆屏衍射中心为泊松亮斑（巴比涅原理的直接体现）。',
            'circular_disk_condition': '泊松亮斑条件: 圆屏半径a与波长λ、距离L满足菲涅耳数Nf > 1'
        },
        'babinet': {
            'title': '巴比涅原理',
            'content': '互补屏的衍射图样互补。即圆屏衍射 = 自由传播 - 圆孔衍射。中心泊松亮斑是巴比涅原理的直接体现。'
        }
    })


# ============================================================
# 主程序
# ============================================================



def main():
    # static/index.html 已作为前端模板就绪
    print("=" * 60)
    print("  光学衍射仿真 — 3D 交互式界面")
    print("=" * 60)
    print()
    print("  访问地址: http://127.0.0.1:8080")
    print()
    print("  操作说明:")
    print("    鼠标拖动: 旋转 3D 视角")
    print("    滚轮:     缩放")
    print("    右键拖动: 平移")
    print()
    print("  快捷键:")
    print("    R: 重置参数")
    print("    F/K: 打开知识库")
    print("    1: 切换到菲涅耳衍射")
    print("    2: 切换到夫琅禾费衍射")
    print()
    print("  按 Ctrl+C 退出")
    print("=" * 60)
    
    # 自动打开浏览器
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:8080')).start()
    
    # 启动 Flask
    app.run(host='127.0.0.1', port=8080, debug=False)


if __name__ == '__main__':
    main()
