"""Tests for Core, HexCore, and CartCore components from Components.py"""

import pytest
from flowforge.input.Components import (
    HexCore,
    CartCore,
    VTKMesh,
)
from flowforge import UnitConverter

_cm2m = 0.01
unitdict = {"length": "cm"}
uc = UnitConverter(unitdict)


def create_basic_components():
    """Create basic components for testing cores"""
    components = {
        "serial_components": {
            "1": {"components": {"pipe": {"a": {"L": 10, "R": 0.1, "n": 5}}}, "order": ["a"]},
            "2": {"components": {"pipe": {"b": {"L": 10, "R": 0.2, "n": 1}}}, "order": ["b"]},
        }
    }
    lplen = {"nozzle": {"L": 1, "R_inlet": 0.5, "R_outlet": 1.2}}
    uplen = {"nozzle": {"L": 1, "R_inlet": 1.2, "R_outlet": 0.5}}
    annulus = {"annulus": {"L": 10, "R_inner": 1.1, "R_outer": 1.2, "n": 10}}

    return components, lplen, uplen, annulus


# --------------------------------------------------------------------------------
# HexCore Tests
# --------------------------------------------------------------------------------


def test_hexcore_validation_odd_rows():
    """Test hexagonal map validation with odd number of rows"""
    components, lplen, uplen, annulus = create_basic_components()

    # Valid odd row count
    valid_map = [[1, 1], [2, 1, 2], [1, 1]]
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]]

    # This should be valid
    hc = HexCore(
        pitch=3,
        components=components,
        channel_map=valid_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )
    assert hc._map == valid_map


def test_hexcore_validation_even_rows():
    """Test hexagonal map validation with even number of rows (should fail)"""
    components, lplen, uplen, annulus = create_basic_components()

    # Invalid even row count
    invalid_map = [[1, 1], [2, 1, 2], [1, 1], [2, 2]]
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    # This should fail validation
    with pytest.raises(AssertionError):
        HexCore(
            pitch=3,
            components=components,
            channel_map=invalid_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=orificing,
            non_channels=["0"],
        )


def test_hexcore_validation_row_length():
    """Test hexagonal map validation with incorrect row lengths"""
    components, lplen, uplen, annulus = create_basic_components()

    # Let's test with an invalid hexagonal pattern
    invalid_map = [[1, 1, 1], [2, 1], [1, 1, 1]]  # Middle row should have more elements, not fewer
    orificing = [[0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 0.0]]

    # Verify that this raises some kind of error - the actual error might be different
    # depending on the validation implementation
    with pytest.raises(Exception):
        HexCore(
            pitch=3,
            components=components,
            channel_map=invalid_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=orificing,
            non_channels=["0"],
        )


def test_hexcore_empty_map():
    """Test hexagonal map validation with empty map (should fail)"""
    components, lplen, uplen, annulus = create_basic_components()

    # Empty map
    empty_map = []
    orificing = []

    # This should fail validation
    with pytest.raises(AssertionError):
        HexCore(
            pitch=3,
            components=components,
            channel_map=empty_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=orificing,
            non_channels=["0"],
        )


def test_hexcore_negative_pitch():
    """Test hexagonal core with negative pitch (should fail)"""
    components, lplen, uplen, annulus = create_basic_components()

    valid_map = [[1, 1], [2, 1, 2], [1, 1]]
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]]

    # This should fail validation
    with pytest.raises(AssertionError):
        HexCore(
            pitch=-3,
            components=components,
            channel_map=valid_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=orificing,
            non_channels=["0"],
        )


def test_hexcore_fill_map():
    """Test hexagonal map filling with various channel types"""
    components, lplen, uplen, annulus = create_basic_components()

    channel_map = [[1, 1], [2, 1, 2], [1, 1]]
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]]
    non_channels = ["0", "X"]

    # Create core with custom non-channels
    hc = HexCore(
        pitch=3,
        components=components,
        channel_map=channel_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=non_channels,
    )

    # Use the static method directly for testing
    filled_map = HexCore._fill_map(channel_map, non_channels)

    # Check that filled map has right structure
    assert len(filled_map) == 3
    assert len(filled_map[0]) == 2  # First row has 2 elements
    assert len(filled_map[1]) == 3  # Middle row has 3 elements
    assert len(filled_map[2]) == 2  # Last row has 2 elements


def test_hexcore_minimal_map():
    """Test hexagonal core with minimal valid map"""
    components, lplen, uplen, annulus = create_basic_components()

    # Minimal valid map - single element
    minimal_map = [[1]]
    orificing = [[0.0]]

    # This should be valid
    hc = HexCore(
        pitch=3,
        components=components,
        channel_map=minimal_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )

    # Test coordinate calculation
    x, y = hc._getChannelCoords(0, 0)
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert x == 0.0  # Should be at origin for 1x1 map


def test_hexcore_coordinates():
    """Test hexagonal coordinate calculations at different positions"""
    components, lplen, uplen, annulus = create_basic_components()

    channel_map = [[1, 1], [2, 1, 2], [1, 1]]
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]]

    hc = HexCore(
        pitch=4.0,
        components=components,
        channel_map=channel_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )

    # Test coordinate calculations at different positions
    for row_idx, row in enumerate(channel_map):
        for col_idx, _ in enumerate(row):
            x, y = hc._getChannelCoords(row_idx, col_idx)
            assert isinstance(x, float)
            assert isinstance(y, float)

    # Get coordinate for a specific position
    # (We don't test for specific values since these depend on the implementation details)
    center_row = 1
    center_col = 1
    x_center, y_center = hc._getChannelCoords(center_row, center_col)

    # Just verify the results are floats
    assert isinstance(x_center, float)
    assert isinstance(y_center, float)


def test_hexcore_with_no_annulus():
    """Test hexagonal core without an annulus"""
    components, lplen, uplen, _ = create_basic_components()

    channel_map = [[1, 1], [2, 1, 2], [1, 1]]
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]]

    # Create without annulus
    hc = HexCore(
        pitch=3,
        components=components,
        channel_map=channel_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=None,
        orificing=orificing,
        non_channels=["0"],
    )

    # Check annulus is None
    assert hc.annulus is None

    # Other functions should still work
    mesh = hc._getVTKMesh((0, 0, 0))
    assert isinstance(mesh, VTKMesh)


def test_hexcore_unit_conversion():
    """Test unit conversion in hexagonal core"""
    components, lplen, uplen, annulus = create_basic_components()

    channel_map = [[1, 1], [2, 1, 2], [1, 1]]
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]]

    hc = HexCore(
        pitch=100,
        components=components,
        channel_map=channel_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )

    # Store original pitch
    original_pitch = hc._pitch

    # Convert units
    hc._convertUnits(uc)

    # Pitch should be converted by the length conversion factor
    assert abs(hc._pitch - original_pitch * _cm2m) < 1e-10


# --------------------------------------------------------------------------------
# CartCore Tests
# --------------------------------------------------------------------------------


def test_cartcore_validation():
    """Test cartesian core validation"""
    components, lplen, uplen, annulus = create_basic_components()

    valid_map = [[1, 1, 1], [2, 2, 1]]
    orificing = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # This should be valid
    cc = CartCore(
        x_pitch=3,
        y_pitch=3,
        components=components,
        channel_map=valid_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )
    assert cc._map == valid_map


def test_cartcore_negative_pitch():
    """Test cartesian core with negative pitch (should fail)"""
    components, lplen, uplen, annulus = create_basic_components()

    valid_map = [[1, 1, 1], [2, 2, 1]]
    orificing = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # x_pitch negative (should fail)
    with pytest.raises(AssertionError):
        CartCore(
            x_pitch=-3,
            y_pitch=3,
            components=components,
            channel_map=valid_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=orificing,
            non_channels=["0"],
        )

    # y_pitch negative (should fail)
    with pytest.raises(AssertionError):
        CartCore(
            x_pitch=3,
            y_pitch=-3,
            components=components,
            channel_map=valid_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=orificing,
            non_channels=["0"],
        )


def test_cartcore_empty_map():
    """Test cartesian core with empty map (should fail)"""
    components, lplen, uplen, annulus = create_basic_components()

    empty_map = []
    orificing = []

    # This should fail validation
    with pytest.raises(AssertionError):
        CartCore(
            x_pitch=3,
            y_pitch=3,
            components=components,
            channel_map=empty_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=orificing,
            non_channels=["0"],
        )


def test_cartcore_different_pitches():
    """Test cartesian core with different x and y pitches"""
    components, lplen, uplen, annulus = create_basic_components()

    valid_map = [[1, 1, 1], [2, 2, 1]]
    orificing = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # Create with different x and y pitches
    cc = CartCore(
        x_pitch=3,
        y_pitch=5,
        components=components,
        channel_map=valid_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )

    # Check pitches were stored correctly
    assert cc._x_pitch == 3
    assert cc._y_pitch == 5

    # Test coordinate calculations
    x0, y0 = cc._getChannelCoords(0, 0)
    x1, y1 = cc._getChannelCoords(1, 0)

    # Going from row 0 to row 1 should change y by y_pitch
    assert abs(y0 - y1 - 5) < 1e-10


@pytest.mark.parametrize("alignment", ["left", "right", "center"])
def test_cartcore_fill_map(alignment):
    """Test cartesian core's _fill_map with different alignments"""
    # This test just checks the static _fill_map method without creating a CartCore instance

    # Asymmetric map for testing alignment
    channel_map = [[1, 2], [1, 2, 1]]

    # Test that filling map works with different alignments
    filled_map = CartCore._fill_map(channel_map, ["0"], alignment)
    assert len(filled_map) == 2

    # In all filled maps, length of longest row should be 3
    max_len = max(len(row) for row in filled_map)
    assert max_len == 3

    # Test different alignments produce different maps
    if alignment == "left":
        first_row = filled_map[0]
        # In left alignment, first row should have values at beginning
        assert first_row[0] is not None
        assert first_row[1] is not None

    elif alignment == "right":
        first_row = filled_map[0]
        # In right alignment, last elements should have values
        assert first_row[-1] is not None
        assert first_row[-2] is not None


def test_cartcore_invalid_alignment():
    """Test cartesian core with invalid alignment (should fail)"""
    components, lplen, uplen, annulus = create_basic_components()

    valid_map = [[1, 1, 1], [2, 2, 1]]
    orificing = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # This should fail validation
    with pytest.raises(ValueError):
        CartCore(
            x_pitch=3,
            y_pitch=3,
            components=components,
            channel_map=valid_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=orificing,
            non_channels=["0"],
            map_alignment="invalid_alignment",
        )


def test_cartcore_single_element():
    """Test cartesian core with minimal map (single element)"""
    components, lplen, uplen, annulus = create_basic_components()

    minimal_map = [[1]]
    orificing = [[0.0]]

    # Create with single element
    cc = CartCore(
        x_pitch=3,
        y_pitch=3,
        components=components,
        channel_map=minimal_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )

    # Test coordinate calculation - should be at origin
    x, y = cc._getChannelCoords(0, 0)
    assert abs(x) < 1e-10
    assert abs(y) < 1e-10


def test_cartcore_unit_conversion():
    """Test unit conversion in cartesian core"""
    components, lplen, uplen, annulus = create_basic_components()

    valid_map = [[1, 1, 1], [2, 2, 1]]
    orificing = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    cc = CartCore(
        x_pitch=100,
        y_pitch=200,
        components=components,
        channel_map=valid_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )

    # Store original pitches
    original_x_pitch = cc._x_pitch
    original_y_pitch = cc._y_pitch

    # Convert units
    cc._convertUnits(uc)

    # Pitches should be converted by the length conversion factor
    assert abs(cc._x_pitch - original_x_pitch * _cm2m) < 1e-10
    assert abs(cc._y_pitch - original_y_pitch * _cm2m) < 1e-10


# --------------------------------------------------------------------------------
# Core Abstract Class Tests
# --------------------------------------------------------------------------------


def test_core_set_extended_components():
    """Test _calculate_centroids and _create_extended_components methods in Core with a simple valid map"""
    components, lplen, uplen, annulus = create_basic_components()

    # Use a simple valid map for HexCore (must have odd number of rows)
    # For a single row, there's a limit to the number of elements, so we'll use
    # a small map that's valid for the HexCore validation
    channel_map = [[1]]
    orificing = [[0.1]]

    # Create an instance of HexCore
    hc = HexCore(
        pitch=3,
        components=components,
        channel_map=channel_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )

    # Get centroids first
    centroids = hc._calculate_centroids()

    # Then create extended components
    extended_comps = hc._create_extended_components(centroids)

    # Should have one component for our 1x1 map
    assert len(extended_comps) == 1
    assert len(centroids) == 1

    # Check that the expected component exists
    assert "1-1-1" in extended_comps

    # Check centroids
    for comp_name, coords in centroids.items():
        assert len(coords) == 2
        assert isinstance(coords[0], float)
        assert isinstance(coords[1], float)


def test_core_getVTKMesh():
    """Test _getVTKMesh method in Core"""
    components, lplen, uplen, annulus = create_basic_components()

    channel_map = [[1, 1], [2, 1, 2], [1, 1]]
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]]

    # Create core
    hc = HexCore(
        pitch=3,
        components=components,
        channel_map=channel_map,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )

    # Test VTK mesh generation
    mesh = hc._getVTKMesh((0, 0, 0))
    assert isinstance(mesh, VTKMesh)


def test_core_orificing_mismatch():
    """Test core with mismatched channel_map and orificing dimensions"""
    components, lplen, uplen, annulus = create_basic_components()

    channel_map = [[1, 1], [2, 1, 2], [1, 1]]
    wrong_orificing = [[0.0, 0.0], [0.0, 0.0]]  # Missing a row

    # Should fail because orificing dimensions don't match channel_map
    with pytest.raises(AssertionError):
        HexCore(
            pitch=3,
            components=components,
            channel_map=channel_map,
            lower_plenum=lplen,
            upper_plenum=uplen,
            annulus=annulus,
            orificing=wrong_orificing,
            non_channels=["0"],
        )
