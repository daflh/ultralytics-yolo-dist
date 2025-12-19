import numpy as np
from scipy.stats import pearsonr, spearmanr
from ultralytics.data.dataset import YOLODataset
import matplotlib.pyplot as plt

ONLY_SPECIFIC_CLASS = None  # e.g., 1 for calculating only 'Car' objects correlation
PLOT_CORRELATION_GRAPH = True


def load_dataset(img_path = None):
    print('Loading dataset...')

    return YOLODataset(
        img_path = img_path,
        augment = False,
        data = {
            'nc': 7
        }
    )


def extract_labels_data(labels):
    labels_min = [] # only store necessary values

    print('Processing labels...')

    for label in labels:
        h, w = label["shape"]

        for cls, bbox, dist in zip(label["cls"], label["bboxes"], label["distances"]):
            if ONLY_SPECIFIC_CLASS is not None and cls.item() != ONLY_SPECIFIC_CLASS:
                continue
        
            x, y, bw, bh = bbox
            dx, dy, dz = dist
            d_euc = (dx**2 + dy**2 + dz**2)**0.5

            # normalize → pixel conversion
            x_px = x * w
            y_px = y * h
            bw_px = bw * w
            bh_px = bh * h

            labels_min.append([ bw_px, bh_px, dz, d_euc ]) # box width, box height, z dist, euclidean dist

    return np.array(labels_min) # N x 4 array


def calculate_correlations(processed_labels):
    combinations = [(0, 2), (0, 3), (1, 2), (1, 3)] # (bw,z), (bw,euc), (bh,z), (bh,euc)
    keys = ['box_width', 'box_height', 'z_distance', 'euclidean_distance']

    print('\nCalculating correlations...')

    for combo in combinations:
        pearson, _ = pearsonr(processed_labels[:, combo[0]], processed_labels[:, combo[1]])
        spearman, _ = spearmanr(processed_labels[:, combo[0]], processed_labels[:, combo[1]])

        print(f"\nCorrelation {keys[combo[0]]} vs {keys[combo[1]]}:")
        print("  Pearson:", pearson)
        print("  Spearman:", spearman)


def fit_inverse_regression(processed_labels, plot_graph=True):
    print("\nFitting inverse regression model for bbox_height vs z-distance...")

    # Extract bbox-height (pixel) and z-distance
    bh = processed_labels[:, 1]
    z = processed_labels[:, 2]

    # Avoid division by extremely small heights
    bh_safe = np.maximum(bh, 1e-6)

    # Inverse term
    inv_bh = 1.0 / bh_safe

    # Fit inverse model: z = a * (1/h) + b
    import time
    time_coeff_start = time.perf_counter()
    coeffs = np.polyfit(inv_bh, z, 1)
    time_coeff_end = time.perf_counter()
    print(f"Fitting time: {(time_coeff_end - time_coeff_start)*1000:.2f} ms")

    a, b = coeffs
    print(f"Inverse model: z ≈ {a:.6f} * (1/bh) + {b:.6f}")

    # Predict z
    z_pred = a * inv_bh + b

    # Calculate MAE
    mae = np.mean(np.abs(z - z_pred))

    # Calculate MRE
    mre = np.mean(np.abs(z - z_pred) / np.maximum(z, 1))

    print(f"\nPrediction Results (Inverse Regression):")
    print(f"MAE: {mae:.4f} meters")
    print(f"MRE: {mre:.4f}")

    if plot_graph:
        # Plot inverse regression curve
        plt.figure(figsize=(8,6))
        plt.scatter(bh, z, s=5, alpha=0.3, label="Data")

        # Draw smooth inverse curve
        bh_line = np.linspace(bh.min(), bh.max(), 300)
        bh_line_safe = np.maximum(bh_line, 1e-6)
        y_line = a * (1.0 / bh_line_safe) + b

        plt.plot(bh_line, y_line, linewidth=2, color="red", label="Inverse Fit")
        plt.xlabel("Bounding Box Height (pixels)")
        plt.ylabel("Z Distance (meters)")
        plt.title("Inverse Regression: bbox_height vs z-distance")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    dataset = load_dataset('D:\\UGM\\tugas akhir\\datasets\\KITTI2017\\images\\train')
    # dataset = load_dataset('D:\\UGM\\tugas akhir\\datasets\\KITTI2017\\images\\val')
    processed_labels = extract_labels_data(dataset.get_labels())

    calculate_correlations(processed_labels)
    fit_inverse_regression(processed_labels, plot_graph=PLOT_CORRELATION_GRAPH)
