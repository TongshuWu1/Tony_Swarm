# estimation.py

import numpy as np

class OdometryEstimation:
    def __init__(self, initial_state=None, noise_std=(0.1, 0.1, 0.1)):
        self.state = np.array(initial_state if initial_state else [0.0, 0.0, 0.0])
        self.noise_std = np.array(noise_std)
        self.covariance = np.diag(self.noise_std) ** 2

    def update(self, v, omega, dt):
        self.state = self.motion_model(self.state, v, omega, dt)
        self.covariance += np.diag(self.noise_std)

    def get_state(self):
        return self.state

    def get_covariance(self):
        return self.covariance

    def motion_model(self, state, v, omega, dt):
        x, y, theta_deg = state
        theta = np.radians(theta_deg)

        # Add elliptical Gaussian noise to the actual robot's motion
        v_noisy_x = v + np.random.normal(0, self.noise_std[0])
        v_noisy_y = v + np.random.normal(0, self.noise_std[1])
        omega_noisy = omega + np.random.normal(0, self.noise_std[2])

        theta += omega_noisy * dt
        x += v_noisy_x * dt * np.cos(theta)
        y += v_noisy_y * dt * np.sin(theta)

        theta_deg = np.degrees(theta) % 360
        return np.array([x, y, theta_deg])