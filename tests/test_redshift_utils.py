from cloelib.utils.redshift import ensure_z_zero_included


def test_ensure_z_zero_included_adds_missing_zero():
    z, order = ensure_z_zero_included(np.array([1.0, 1.2, 1.4, 1.65]))
    assert_array_equal(z, [0.0, 1.0, 1.2, 1.4, 1.65])
    # z is already ascending, so the descending-to-z order is a plain reversal
    assert_array_equal(order, [4, 3, 2, 1, 0])


def test_ensure_z_zero_included_order_reorders_descending_camb_output():
    """`order` must map a quantity camb always returns sorted by descending
    z (regardless of input order) onto `z`'s own order, even when `z` is
    left unsorted (the `z_min <= 0.001` branch doesn't sort its input)."""
    redshifts = np.array([1.4, 1.0, 1.65, 1.2, 0.0])
    z, order = ensure_z_zero_included(redshifts)
    assert_array_equal(z, redshifts)

    camb_descending_z = np.sort(z)[::-1]
    camb_quantity = np.arange(len(z))[::-1]  # highest value at z=0, decreasing with z
    reordered = camb_quantity[order]

    for zi, value in zip(z, reordered):
        expected_value = camb_quantity[np.where(camb_descending_z == zi)[0][0]]
        assert value == expected_value
