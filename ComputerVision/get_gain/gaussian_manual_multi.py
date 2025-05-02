from matplotlib import pyplot as plt
import numpy as np
import re

# === Read points ===121
def read_tuples_from_file(filename):
    tuples_list = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                tuples = re.findall(r'\(([^)]+)\)', line)
                for tpl in tuples:
                    a, b = map(int, tpl.split(','))
                    tuples_list.append((a, b))  # <-- no filter
    except FileNotFoundError:
        print(f"File not found: {filename}")
    return tuples_list

# === Calculate Gaussian ===
def calculate_mean_cov(points):
    data = np.array(points)
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mean.tolist(), cov.tolist()

# === Plot Ellipse ===
def plot_gaussian_ellipse(mean, cov, ax, edgecolor='black'):
    from matplotlib.patches import Ellipse
    import math

    vals, vecs = np.linalg.eigh(np.array(cov))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)

    ell = Ellipse(xy=mean, width=width*2, height=height*2, angle=theta,
                  edgecolor=edgecolor, fc='None', lw=2)
    ax.add_patch(ell)

# === Plot stats and Print ===
def plot_stats(points, ax, plot_summary=True, edgecolor='black', epsilon=0.2):
    mean, cov = calculate_mean_cov(points)

    # Add looseness
    cov[0][0] += epsilon
    cov[1][1] += epsilon

    cov_np = np.array(cov)
    inv_cov_np = np.linalg.inv(cov_np)

    mean_list = [float(v) for v in mean]
    inv_cov_list = [[float(val) for val in row] for row in inv_cov_np.tolist()]

    # === Nice print ===
    print("Mean and Inverted Covariance:", mean_list, ",", inv_cov_list)
    print()

    # Plot Gaussian
    plot_gaussian_ellipse(mean_list, cov, ax, edgecolor=edgecolor)

    # Plot points
    if plot_summary:
        ax.scatter([p[0] for p in points], [p[1] for p in points], color=edgecolor, alpha=0.5)

# === Main ===
if __name__ == "__main__":
    red_left = read_tuples_from_file("colors/red_left.txt")
    red_center = read_tuples_from_file("colors/red_center.txt")
    red_right = read_tuples_from_file("colors/red_right.txt")

    colors = [
        ("red_left", red_left, 'yellow'),
        ("red_center", red_center, 'red'),
        ("red_right", red_right, 'purple')
    ]

    fig, ax = plt.subplots()

    for name, points, c in colors:
        print("-------", name)
        if len(points) == 0:
            continue
        plot_stats(points, ax, plot_summary=True, edgecolor=c, epsilon=0.2)

    plt.xlabel("B")
    plt.ylabel("A")
    ax.grid()
    plt.tight_layout()
    plt.show()
