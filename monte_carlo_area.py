import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def find_intersections(a1, b1, c1, a2, b2, c2):
    # f(x) - g(x) = (a1-a2)x^2 + (b1-b2)x + (c1-c2) = 0
    A = a1 - a2
    B = b1 - b2
    C = c1 - c2
    
    if A == 0:
        if B == 0:
            return None # 完全に一致
        else:
            return [-C / B] # 1次方程式になる場合
            
    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        return [] # 交点なし
    elif discriminant == 0:
        return [-B / (2*A)] # 接する
    else:
        x1 = (-B - np.sqrt(discriminant)) / (2*A)
        x2 = (-B + np.sqrt(discriminant)) / (2*A)
        return sorted([x1, x2])

def calculate_area_integration(a1, b1, c1, a2, b2, c2, x_start, x_end):
    # 積分する関数: |f(x) - g(x)|
    def diff(x):
        return abs(quadratic(x, a1, b1, c1) - quadratic(x, a2, b2, c2))
    
    area, error = quad(diff, x_start, x_end)
    return area

def calculate_area_monte_carlo(a1, b1, c1, a2, b2, c2, x_start, x_end, num_samples=100000):
    # サンプリング領域（バウンディングボックス）の決定
    x_vals = np.linspace(x_start, x_end, 1000)
    y1_vals = quadratic(x_vals, a1, b1, c1)
    y2_vals = quadratic(x_vals, a2, b2, c2)
    
    ymin = min(np.min(y1_vals), np.min(y2_vals))
    ymax = max(np.max(y1_vals), np.max(y2_vals))
    
    # 領域が潰れないようにマージンを設ける（境界ギリギリだと判定エラーのリスクがあるため）
    y_margin = (ymax - ymin) * 0.1
    if y_margin == 0:
        y_margin = 1.0
    ymin -= y_margin
    ymax += y_margin
    
    box_area = (x_end - x_start) * (ymax - ymin)
    
    # 乱数で点を生成
    rand_x = np.random.uniform(x_start, x_end, num_samples)
    rand_y = np.random.uniform(ymin, ymax, num_samples)
    
    # 点が2つの曲線の間にあるか判定
    y_curves1 = quadratic(rand_x, a1, b1, c1)
    y_curves2 = quadratic(rand_x, a2, b2, c2)
    
    upper_curve = np.maximum(y_curves1, y_curves2)
    lower_curve = np.minimum(y_curves1, y_curves2)
    
    inside = (rand_y >= lower_curve) & (rand_y <= upper_curve)
    points_inside = np.sum(inside)
    
    # モンテカルロ法による面積計算
    mc_area = (points_inside / num_samples) * box_area
    
    return mc_area, rand_x, rand_y, inside

def main():
    print("任意の2つの二次関数 f(x) = a1*x^2 + b1*x + c1 と g(x) = a2*x^2 + b2*x + c2 の間の面積を求めます。")
    # ここでは例として以下の関数を使用します。任意に変更可能です。
    # f(x) = -x^2 + 4
    # g(x) = x^2 - 2x
    a1, b1, c1 = -1, 0, 4
    a2, b2, c2 = 1, -2, 0
    
    print(f"\n関数1: f(x) = {a1}x^2 + {b1}x + {c1}")
    print(f"関数2: g(x) = {a2}x^2 + {b2}x + {c2}")
    
    intersections = find_intersections(a1, b1, c1, a2, b2, c2)
    
    if not intersections or len(intersections) < 2:
        print("2つの関数は2点で交わらないため、囲まれる閉じた領域がありません。")
        return
        
    x_start, x_end = intersections[0], intersections[1]
    print(f"\n交点: x1 = {x_start:.4f}, x2 = {x_end:.4f}")
    
    # 定積分による理論値の計算
    area_int = calculate_area_integration(a1, b1, c1, a2, b2, c2, x_start, x_end)
    print(f"\n定積分による面積: {area_int:.6f}")
    
    # モンテカルロ法による面積の計算
    num_samples = 100000
    area_mc, rand_x, rand_y, inside = calculate_area_monte_carlo(a1, b1, c1, a2, b2, c2, x_start, x_end, num_samples)
    print(f"モンテカルロ法による面積 (N={num_samples}): {area_mc:.6f}")
    
    # 比較
    error = abs(area_int - area_mc)
    accuracy = (1 - error / area_int) * 100 if area_int != 0 else 0
    print(f"\n絶対誤差: {error:.6f}")
    print(f"精度: {accuracy:.4f}%")
    
    # 結果の可視化
    plt.figure(figsize=(10, 6))
    
    # サンプリング点をプロット (多すぎると重いので表示用に少し間引く等の工夫も可能ですが、今回はそのまま表示)
    plt.scatter(rand_x[inside], rand_y[inside], color='blue', s=1, alpha=0.1, label='Inside')
    plt.scatter(rand_x[~inside], rand_y[~inside], color='red', s=1, alpha=0.1, label='Outside')
    
    # 関数をプロット
    x_vals = np.linspace(x_start - 1, x_end + 1, 400)
    y1_vals = quadratic(x_vals, a1, b1, c1)
    y2_vals = quadratic(x_vals, a2, b2, c2)
    
    plt.plot(x_vals, y1_vals, 'k-', linewidth=2, label='f(x)')
    plt.plot(x_vals, y2_vals, 'g-', linewidth=2, label='g(x)')
    
    # 交点の境界線を引く
    plt.axvline(x=x_start, color='gray', linestyle='--')
    plt.axvline(x=x_end, color='gray', linestyle='--')
    
    plt.title('Monte Carlo Integration of Area Between Two Quadratic Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # 画像として保存
    plt.savefig('monte_carlo_result.png')
    print("\nグラフを 'monte_carlo_result.png' に保存しました。")

if __name__ == "__main__":
    main()
