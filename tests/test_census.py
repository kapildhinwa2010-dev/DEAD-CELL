import numpy as np
import pytest

from dead_cell.census import dead_cells, total_waste, weight_report


def _w():
    w = np.ones((4, 4))
    w[0, :] = 0.0
    w[:, 2] = 0.0
    return w


def test_weight_report():
    r = weight_report(_w(), name="dense")
    assert r["name"] == "dense"
    assert r["shape"] == [4, 4]
    assert r["params"] == 16
    assert r["zero_rows"] == [0]
    assert r["zero_cols"] == [2]
    assert r["zero_params"] == 7
    assert r["waste_fraction"] == pytest.approx(7.0 / 16.0)


def test_all_zero_matrix():
    r = weight_report(np.zeros((3, 5)))
    assert r["zero_rows"] == [0, 1, 2]
    assert r["zero_cols"] == [0, 1, 2, 3, 4]
    assert r["waste_fraction"] == 1.0


def test_healthy_matrix():
    rng = np.random.default_rng(0)
    r = weight_report(rng.normal(size=(6, 6)))
    assert r["zero_rows"] == []
    assert r["zero_cols"] == []
    assert r["waste_fraction"] == 0.0


def test_dead_cells_with_activations():
    rng = np.random.default_rng(1)
    acts = rng.normal(size=(5, 4)) + 3.0
    acts[:, 1] = 0.0
    entries = dead_cells([np.eye(4)], activations=[acts])
    assert entries[0]["never_active_neurons"] == [1]
    assert entries[0]["occupancy_rate"] == pytest.approx(0.75)


def test_dead_cells_without_activations():
    entries = dead_cells([np.eye(4), _w()])
    assert len(entries) == 2
    assert entries[0]["name"] == "layer_0"
    assert "never_active_neurons" not in entries[0]
    assert entries[1]["zero_params"] == 7


def test_total_waste():
    entries = dead_cells([np.eye(4), _w()])
    t = total_waste(entries)
    assert t["num_layers"] == 2
    assert t["total_params"] == 32
    assert t["total_zero_params"] == 7
    assert t["waste_fraction"] == pytest.approx(7.0 / 32.0)


def test_errors():
    with pytest.raises(ValueError):
        dead_cells([])
    with pytest.raises(ValueError):
        dead_cells([np.eye(4)], activations=[])
    with pytest.raises(ValueError):
        dead_cells([np.eye(4)], activations=[np.zeros((2, 3))])
    with pytest.raises(ValueError):
        weight_report(np.zeros(4))
    with pytest.raises(ValueError):
        total_waste([])