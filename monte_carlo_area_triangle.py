import numpy as np
import matplotlib.pyplot as plt

def linear(x, m, c):
    """一次関数 y = mx + c"""
    return m * x + c

def get_intersection(m1, c1, m2, c2):
    """2つの一次関数の交点を求める"""
    if m1 == m2:
        return None # 平行
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

def triangle_area(v1, v2, v3):
    """3点の座標から三角形の面積を計算する（理論値）"""
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def is_inside_triangle(px, py, v1, v2, v3):
    """点(px, py)がv1, v2, v3からなる三角形の内部にあるか判定する"""
    def sign(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
    
    b1 = sign(px, py, v1[0], v1[1], v2[0], v2[1]) < 0.0
    b2 = sign(px, py, v2[0], v2[1], v3[0], v3[1]) < 0.0
    b3 = sign(px, py, v3[0], v3[1], v1[0], v1[1]) < 0.0
    
    return (b1 == b2) & (b2 == b3)

def calculate_area_monte_carlo(v1, v2, v3, num_samples=100000):
    """モンテカルロ法で三角形の面積を計算する"""
    x_coords = [v1[0], v2[0], v3[0]]
    y_coords = [v1[1], v2[1], v3[1]]
    
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    # バウンディングボックスにマージンを持たせる
    x_margin = (xmax - xmin) * 0.1
    y_margin = (ymax - ymin) * 0.1
    if x_margin == 0: x_margin = 1.0
    if y_margin == 0: y_margin = 1.0
    
    xmin -= x_margin
    xmax += x_margin
    ymin -= y_margin
    ymax += y_margin
    
    box_area = (xmax - xmin) * (ymax - ymin)
    
    # ランダムな点を生成
    rand_x = np.random.uniform(xmin, xmax, num_samples)
    rand_y = np.random.uniform(ymin, ymax, num_samples)
    
    # 内外判定
    inside = is_inside_triangle(rand_x, rand_y, v1, v2, v3)
    points_inside = np.sum(inside)
    
    mc_area = (points_inside / num_samples) * box_area
    
    return mc_area, rand_x, rand_y, inside, (xmin, xmax, ymin, ymax)

def main():
    print("任意の3つの一次関数で囲まれる面積を求めます。")
    # L1: y = m1*x + c1
    # L2: y = m2*x + c2
    # L3: y = m3*x + c3
    m1, c1 = 1.0, 0.0    # y = x
    m2, c2 = -1.0, 4.0   # y = -x + 4
    m3, c3 = 0.0, 0.0    # y = 0
    
    print(f"関数1: y = {m1}x + {c1}")
    print(f"関数2: y = {m2}x + {c2}")
    print(f"関数3: y = {m3}x + {c3}")
    
    # 交点を求める
    p12 = get_intersection(m1, c1, m2, c2)
    p23 = get_intersection(m2, c2, m3, c3)
    p31 = get_intersection(m3, c3, m1, c1)
    
    if not p12 or not p23 or not p31:
        print("平行な直線があるため、閉じた三角形を形成しません。")
        return
        
    if p12 == p23 or p23 == p31 or p31 == p12:
        print("3つの直線が1点で交わるため、面積は0です。")
        return
        
    print(f"\n頂点1: {p12}")
    print(f"頂点2: {p23}")
    print(f"頂点3: {p31}")
    
    # 理論値（幾何学的計算）
    area_exact = triangle_area(p12, p23, p31)
    print(f"\n理論値による面積: {area_exact:.6f}")
    
    # モンテカルロ法
    num_samples = 100000
    area_mc, rand_x, rand_y, inside, bbox = calculate_area_monte_carlo(p12, p23, p31, num_samples)
    print(f"モンテカルロ法による面積 (N={num_samples}): {area_mc:.6f}")
    
    # 精度計算
    error = abs(area_exact - area_mc)
    accuracy = (1 - error / area_exact) * 100 if area_exact != 0 else 0
    print(f"\n絶対誤差: {error:.6f}")
    print(f"精度: {accuracy:.4f}%")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    
    # プロット点
    plt.scatter(rand_x[inside], rand_y[inside], color='blue', s=1, alpha=0.1, label='Inside')
    plt.scatter(rand_x[~inside], rand_y[~inside], color='red', s=1, alpha=0.1, label='Outside')
    
    # 直線をプロット
    xmin, xmax, ymin, ymax = bbox
    x_vals = np.linspace(xmin, xmax, 400)
    
    plt.plot(x_vals, linear(x_vals, m1, c1), 'k-', linewidth=2, label='L1')
    plt.plot(x_vals, linear(x_vals, m2, c2), 'g-', linewidth=2, label='L2')
    plt.plot(x_vals, linear(x_vals, m3, c3), 'm-', linewidth=2, label='L3')
    
    # 頂点をプロット
    vertices_x = [p12[0], p23[0], p31[0]]
    vertices_y = [p12[1], p23[1], p31[1]]
    plt.plot(vertices_x, vertices_y, 'ro', markersize=8, label='Vertices')
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('Monte Carlo Integration of Triangle Area')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('monte_carlo_triangle_result.png')
    print("\nグラフを 'monte_carlo_triangle_result.png' に保存しました。")

if __name__ == "__main__":
    main()
