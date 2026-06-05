import numpy as np


class ParticleFilter:
    def __init__(self, n_particles, tag_map, room_bounds):
        """
        n_particles : int — number of particles
        tag_map     : list of dicts [{'x': float, 'y': float}, ...]
        room_bounds : dict  {'x_min', 'x_max', 'y_min', 'y_max'}
        """
        self.n_particles = n_particles
        self.tag_map = tag_map
        self.room_bounds = room_bounds
        self.particles = None  # shape (N, 3): x, y, theta
        self.weights = None    # shape (N,)

    # Uniform random distribution over room bounds
    def initialize(self):
        """Uniform random distribution over room bounds."""
        x = np.random.uniform(
            self.room_bounds["x_min"], self.room_bounds["x_max"], self.n_particles
        )
        y = np.random.uniform(
            self.room_bounds["y_min"], self.room_bounds["y_max"], self.n_particles
        )
        theta = np.random.uniform(-np.pi, np.pi, self.n_particles)
        self.particles = np.column_stack([x, y, theta])
        self.weights = np.ones(self.n_particles) / self.n_particles

    # Prediction phase implementing the odometry motion model
    def predict(self, delta_x, delta_y, delta_theta, alphas):
        """
        Apply odometry delta to every particle with Gaussian noise.

        alphas : [a1, a2, a3, a4]
            a1, a2 scale translation noise
            a3, a4 scale rotation noise
        """
        translation = np.sqrt(delta_x ** 2 + delta_y ** 2)
        rotation = delta_theta

        sigma_trans = alphas[0] * translation + alphas[1] * abs(rotation)
        sigma_rot = alphas[2] * abs(rotation) + alphas[3] * translation

        noise_trans = np.random.normal(0.0, max(sigma_trans, 1e-6), self.n_particles)
        noise_rot = np.random.normal(0.0, max(sigma_rot, 1e-6), self.n_particles)

        effective_trans = translation + noise_trans
        effective_rot = rotation + noise_rot

        self.particles[:, 0] += effective_trans * np.cos(self.particles[:, 2])
        self.particles[:, 1] += effective_trans * np.sin(self.particles[:, 2])
        self.particles[:, 2] += effective_rot
        self.particles[:, 2] = self._wrap_angle(self.particles[:, 2])

    # Measurement update implementing the multi-hypothesis sensor model
    def update(self, observation, sigma_dist=0.3, sigma_bearing=0.2):
        """
        Multi-hypothesis sensor model.

        p(z | x_particle) = sum_{i=1}^{N_tags} p(z | x_particle, tag_i)

        observation : (distance, bearing) — both in robot frame
                      distance  [metres]
                      bearing   [radians]
        """
        obs_dist, obs_bearing = observation
        likelihoods = np.zeros(self.n_particles)

        for tag in self.tag_map:
            dx = tag["x"] - self.particles[:, 0]
            dy = tag["y"] - self.particles[:, 1]

            expected_dist = np.sqrt(dx ** 2 + dy ** 2)
            expected_bearing = self._wrap_angle(
                np.arctan2(dy, dx) - self.particles[:, 2]
            )

            dist_err = obs_dist - expected_dist
            bearing_err = self._wrap_angle(obs_bearing - expected_bearing)

            likelihood = np.exp(
                -0.5 * dist_err ** 2 / sigma_dist ** 2
            ) * np.exp(
                -0.5 * bearing_err ** 2 / sigma_bearing ** 2
            )
            likelihoods += likelihood

        self.weights *= likelihoods
        total = self.weights.sum()

        if total < 1e-10:
            # Weight collapse: reset to uniform to survive
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights /= total

    # Selection process to multiply high-weight particles and discard low-weight ones
    def resample(self):
        """
        Systematic (low-variance) resampling.
        Only call when n_eff() drops below a threshold, e.g. N/2.
        """
        n = self.n_particles
        positions = (np.arange(n) + np.random.uniform(0.0, 1.0)) / n
        cumsum = np.cumsum(self.weights)
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n

    # Calculation of effective particle sample size to monitor degeneration
    def n_eff(self):
        """Effective particle count. Low value → resample."""
        return 1.0 / np.sum(self.weights ** 2)

    # Computation of the final state vector using weighted circular mean
    def estimate(self):
        """
        Weighted mean pose.
        Theta uses circular mean to handle angle wrap-around.
        Returns (x, y, theta).
        """
        x = float(np.sum(self.weights * self.particles[:, 0]))
        y = float(np.sum(self.weights * self.particles[:, 1]))
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta = float(np.arctan2(sin_sum, cos_sum))
        return x, y, theta

    # Normalization helper to map values back to valid angular space
    @staticmethod
    def _wrap_angle(angle):
        """Wrap angle(s) to [-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
