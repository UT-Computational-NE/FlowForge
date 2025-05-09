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
    """Test getVTKMesh method in Core"""
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
    mesh = hc.getVTKMesh((0, 0, 0))
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
