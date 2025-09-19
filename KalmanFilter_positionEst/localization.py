import numpy as np
import matplotlib.patches as patches

class OdometryEstimator:
    def __init__(self, init_pos, init_cov):
        """
        init_pos: [x, y, angle_in_degrees]
        init_cov: Initial covariance matrix (3x3)
        """
        self.mu = np.array(init_pos, dtype=float)
        self.cov = np.array(init_cov, dtype=float)

    def predict(self, u, dt, motion_noise_cov):
        """
        Predict next state and rotate motion uncertainty to world frame.
        """
        v, omega = u
        theta = np.radians(self.mu[2])

        delta_x = v * np.cos(theta) * dt
        delta_y = v * np.sin(theta) * dt
        delta_theta = omega * dt

        self.mu += np.array([delta_x, delta_y, np.degrees(delta_theta)])
        self.mu[2] %= 360

        # Jacobian of motion model w.r.t. state
        G = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0,  1]
        ])

        # === 🌀 Rotate local noise into world frame ===
        R_theta = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        rotated_motion_noise = R_theta @ motion_noise_cov @ R_theta.T

        # EKF Covariance Update
        self.cov = G @ self.cov @ G.T + rotated_motion_noise


    def correct(self, measurements, measurement_noise_cov, alpha=1.0):
        """
        Perform EKF correction step.

        measurements: list of (range_meas, bearing_meas, landmark_x, landmark_y)
        measurement_noise_cov: 2x2 covariance for [range, bearing]
        alpha: optional blending factor for correction (default 1.0 = full correction)
        """
        for z_r, z_b, lm_x, lm_y in measurements:
            dx = lm_x - self.mu[0]
            dy = lm_y - self.mu[1]
            q = dx**2 + dy**2

            if q < 1e-6:
                continue  # avoid division by zero

            sqrt_q = np.sqrt(q)

            # Expected measurement [range, bearing]
            z_hat = np.array([
                sqrt_q,
                (np.degrees(np.arctan2(dy, dx)) - self.mu[2]) % 360
            ])
            if z_hat[1] > 180:
                z_hat[1] -= 360

            # Jacobian H
            H = np.array([
                [-dx / sqrt_q, -dy / sqrt_q, 0],
                [dy / q, -dx / q, -1]
            ])

            # Kalman Gain
            S = H @ self.cov @ H.T + measurement_noise_cov
            K = self.cov @ H.T @ np.linalg.inv(S)

            # Innovation (measurement residual)
            z = np.array([z_r, z_b])
            y = z - z_hat
            if y[1] > 180:
                y[1] -= 360
            elif y[1] < -180:
                y[1] += 360

            # Correction with alpha (optional blending)
            self.mu += alpha * (K @ y)
            self.mu[2] %= 360

            # Covariance update
            I = np.eye(3)
            self.cov = (I - K @ H) @ self.cov

    def draw_uncertainty_ellipse(self, ax, n_std=2.0, **kwargs):
        """
        Draw the 2D uncertainty ellipse from x-y covariance.
        """
        cov_xy = self.cov[:2, :2]
        mean_xy = self.mu[:2]

        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        eigvals = np.maximum(eigvals, 1e-10)  # prevent collapse
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigvals)

        ellipse = patches.Ellipse(mean_xy, width, height,
                                  angle=angle, zorder=5, **kwargs)
        ax.add_patch(ellipse)
