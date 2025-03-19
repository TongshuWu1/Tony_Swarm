# estimation.py

import numpy as np
class OdometryEstimation:
    def __init__(self, initial_state=None, noise_std=(0.1, 0.1, 0.05)):
        self.state = np.array(initial_state if initial_state else [0.0, 0.0, 0.0])
        self.noise_std = np.array(noise_std)
        self.covariance = np.diag(self.noise_std) ** 2

    def update(self, v, omega, dt):
        predicted_state = self.motion_model(self.state, v, omega, dt)
        noise = np.random.normal(0, self.noise_std)
        self.state = predicted_state + noise
        self.covariance += np.diag(self.noise_std) ** 2

    def get_state(self):
        return self.state

    def get_covariance(self):
        return self.covariance

    def motion_model(self, state, v, omega, dt):
        x, y, theta = state
        theta += omega * dt
        x += v * dt * np.cos(theta)
        y += v * dt * np.sin(theta)
        return np.array([x, y, theta])