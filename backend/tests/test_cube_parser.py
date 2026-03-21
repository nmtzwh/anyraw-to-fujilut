"""Tests for .cube 3D LUT file parser."""
import io
import numpy as np
import pytest

from backend.cube_parser import parse_cube


class TestParseCube:
    def _basic_cube_content(self) -> bytes:
        return b"""TITLE "Test LUT"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""

    def test_parse_cube_basic_shape_and_values(self):
        lut = parse_cube(io.BytesIO(self._basic_cube_content()))
        assert lut.shape == (2, 2, 2, 3)
        assert lut.dtype == np.float32
        # Verify specific corner values
        assert np.allclose(lut[0, 0, 0], [0.0, 0.0, 0.0])
        assert np.allclose(lut[1, 1, 1], [1.0, 1.0, 1.0])
        assert np.allclose(lut[1, 0, 0], [1.0, 0.0, 0.0])
        assert np.allclose(lut[0, 1, 0], [0.0, 1.0, 0.0])
        assert np.allclose(lut[0, 0, 1], [0.0, 0.0, 1.0])

    def test_parse_cube_strips_comments(self):
        content = b"""# This is a comment
TITLE "My LUT"
LUT_3D_SIZE 2
# Another comment
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""
        lut = parse_cube(io.BytesIO(content))
        assert lut.shape == (2, 2, 2, 3)

    def test_parse_cube_33_point(self):
        # 33-point LUT: 33^3 = 35937 values
        header = b"LUT_3D_SIZE 33\n"
        values = b"0.0 0.0 0.0\n" * 35937
        lut = parse_cube(io.BytesIO(header + values))
        assert lut.shape == (33, 33, 33, 3)
        assert lut.dtype == np.float32

    def test_parse_cube_3_point_identity(self):
        # 3-point identity LUT: all corners map to themselves
        header = b"LUT_3D_SIZE 3\n"
        values = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    values.append(f"{i/2:.6f} {j/2:.6f} {k/2:.6f}")
        content = header + "\n".join(values).encode()
        lut = parse_cube(io.BytesIO(content))
        assert lut.shape == (3, 3, 3, 3)
        # The LUT is an identity mapping
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    expected = [i / 2.0, j / 2.0, k / 2.0]
                    assert np.allclose(lut[i, j, k], expected), f"Failed at ({i},{j},{k})"

    def test_parse_cube_rejects_wrong_value_count(self):
        header = b"LUT_3D_SIZE 3\n"  # expects 27 values
        values = b"0.0 0.0 0.0\n" * 10  # only 10 values
        with pytest.raises(ValueError, match="values"):
            parse_cube(io.BytesIO(header + values))

    def test_parse_cube_empty_lines_ignored(self):
        content = b"""LUT_3D_SIZE 2

0.0 0.0 0.0

1.0 0.0 0.0

0.0 1.0 0.0

1.0 1.0 0.0

0.0 0.0 1.0

1.0 0.0 1.0

0.0 1.0 1.0

1.0 1.0 1.0
"""
        lut = parse_cube(io.BytesIO(content))
        assert lut.shape == (2, 2, 2, 3)
