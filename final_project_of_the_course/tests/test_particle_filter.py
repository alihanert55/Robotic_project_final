import numpy as np
import pytest
from particle_filter import ParticleFilter

# Environment geometry and configurations shared across tests
TAG_MAP = [
    {"x": -0.274, "y":  1.280},
    {"x":  0.263, "y":  1.280},
    {"x": -1.240, "y": -0.719},
    {"x": -1.240, "y":  0.698},
    {"x":  1.280, "y": -0.747},
    {"x":  1.280, "y":  0.468},
    {"x":  0.647, "y": -1.241},
    {"x": -0.940, "y": -1.240},
]

ROOM_BOUNDS = {"x_min": -1.5, "x_max": 1.5, "y_min": -1.5, "y_max": 1.5}
N = 500


@pytest.fixture
def pf():
    f = ParticleFilter(N, TAG_MAP, ROOM_BOUNDS)
    f.initialize()
    return f


# Verification of the initial state generation
class TestInitialize:
    def test_particle_shape(self, pf):
        assert pf.particles.shape == (N, 3)

    def test_weight_shape(self, pf):
        assert pf.weights.shape == (N,)

    def test_weights_sum_to_one(self, pf):
        assert pytest.approx(pf.weights.sum(), abs=1e-9) == 1.0

    def test_particles_within_bounds(self, pf):
        assert np.all(pf.particles[:, 0] >= ROOM_BOUNDS["x_min"])
        assert np.all(pf.particles[:, 0] <= ROOM_BOUNDS["x_max"])
        assert np.all(pf.particles[:, 1] >= ROOM_BOUNDS["y_min"])
        assert np.all(pf.particles[:, 1] <= ROOM_BOUNDS["y_max"])

    def test_theta_within_range(self, pf):
        assert np.all(pf.particles[:, 2] >= -np.pi)
        assert np.all(pf.particles[:, 2] <= np.pi)


# Verification of the motion model and state propagation
class TestPredict:
    def test_particles_move(self, pf):
        before = pf.particles.copy()
        alphas = [0.1, 0.1, 0.1, 0.1]
        pf.predict(0.2, 0.0, 0.1, alphas)
        # Not all particles should be identical to before
        assert not np.allclose(pf.particles, before)

    def test_particle_count_unchanged(self, pf):
        alphas = [0.1, 0.1, 0.1, 0.1]
        pf.predict(0.1, 0.0, 0.0, alphas)
        assert pf.particles.shape == (N, 3)

    def test_theta_stays_wrapped(self, pf):
        alphas = [0.05, 0.05, 0.05, 0.05]
        # Force a large rotation to stress wrapping
        pf.predict(0.0, 0.0, 3.0, alphas)
        assert np.all(pf.particles[:, 2] >= -np.pi)
        assert np.all(pf.particles[:, 2] <= np.pi)

    def test_zero_delta_small_spread(self, pf):
        """Zero delta → particles should barely move (only noise)."""
        before = pf.particles.copy()
        alphas = [0.01, 0.01, 0.01, 0.01]
        pf.predict(0.0, 0.0, 0.0, alphas)
        max_shift = np.abs(pf.particles - before).max()
        assert max_shift < 0.5


# Verification of the measurement model and weight adjustment
class TestUpdate:
    def test_weights_sum_to_one_after_update(self, pf):
        pf.update((1.0, 0.3))
        assert pytest.approx(pf.weights.sum(), abs=1e-9) == 1.0

    def test_weights_change_after_update(self, pf):
        before = pf.weights.copy()
        pf.update((1.0, 0.3))
        assert not np.allclose(pf.weights, before)

    def test_weights_nonnegative(self, pf):
        pf.update((0.8, -0.2))
        assert np.all(pf.weights >= 0.0)

    def test_weight_collapse_recovery(self, pf):
        """Wildly wrong observation → weights should not all be zero."""
        pf.update((999.0, 999.0))
        assert pytest.approx(pf.weights.sum(), abs=1e-9) == 1.0


# Verification of the resampling process and particle survival
class TestResample:
    def test_particle_count_preserved(self, pf):
        pf.update((1.0, 0.2))
        pf.resample()
        assert pf.particles.shape == (N, 3)

    def test_weights_uniform_after_resample(self, pf):
        pf.update((1.0, 0.2))
        pf.resample()
        expected = 1.0 / N
        assert np.allclose(pf.weights, expected)

    def test_resampled_particles_come_from_original(self, pf):
        """Every resampled particle must be one that existed before."""
        pf.update((1.0, 0.2))
        before = set(map(tuple, pf.particles.tolist()))
        pf.resample()
        after = set(map(tuple, pf.particles.tolist()))
        assert after.issubset(before)


# Verification of effective particle sample size calculations
class TestNEff:
    def test_n_eff_uniform_weights(self, pf):
        # Uniform weights → n_eff should equal N
        assert pytest.approx(pf.n_eff(), rel=1e-6) == float(N)

    def test_n_eff_drops_after_update(self, pf):
        before = pf.n_eff()
        pf.update((1.0, 0.3))
        assert pf.n_eff() <= before


# Verification of the final state estimation matrix
class TestEstimate:
    def test_estimate_returns_three_values(self, pf):
        x, y, theta = pf.estimate()
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(theta, float)

    def test_estimate_within_room(self, pf):
        x, y, _ = pf.estimate()
        assert ROOM_BOUNDS["x_min"] <= x <= ROOM_BOUNDS["x_max"]
        assert ROOM_BOUNDS["y_min"] <= y <= ROOM_BOUNDS["y_max"]

    def test_estimate_theta_wrapped(self, pf):
        _, _, theta = pf.estimate()
        assert -np.pi <= theta <= np.pi

    def test_estimate_single_particle(self):
        """With one particle, estimate must match that particle exactly."""
        pf = ParticleFilter(1, TAG_MAP, ROOM_BOUNDS)
        pf.initialize()
        pf.particles[0] = [0.5, -0.3, 1.2]
        pf.weights[0] = 1.0
        x, y, theta = pf.estimate()
        assert pytest.approx(x) == 0.5
        assert pytest.approx(y) == -0.3
        assert pytest.approx(theta) == 1.2


# Verification of the angular normalization utility
class TestWrapAngle:
    @pytest.mark.parametrize("angle,expected", [
        (0.0, 0.0),
        (2 * np.pi, 0.0),
        (2 * np.pi, 0.0),
        (0.5, 0.5),
        (-0.5, -0.5),
    ])
    def test_wrap_scalar(self, angle, expected):
        result = ParticleFilter._wrap_angle(np.array(angle))
        assert pytest.approx(float(result), abs=1e-9) == expected

    @pytest.mark.parametrize("angle", [np.pi, -np.pi, 3 * np.pi])
    def test_wrap_pi_boundary(self, angle):
        """pi and -pi are the same angle — both are valid outputs."""
        result = float(ParticleFilter._wrap_angle(np.array(angle)))
        assert pytest.approx(abs(result), abs=1e-9) == np.pi
