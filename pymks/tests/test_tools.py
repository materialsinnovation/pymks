from pymks.tools import _get_ticks_params


def test_get_ticks_params():
    l = 4
    result = ([0, 1, 2, 3, 4], [-2, -1, 0, 1, 2])
    assert result == _get_ticks_params(l)
