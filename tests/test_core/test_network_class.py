"""
Testing the functionality of the GRN class
"""

# External Libraries
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# Local imports to test
from gene_reg_net.core import GRN


@pytest.fixture
def grn_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["a", "b", "c", "d", "b"],
            "target": ["b", "c", "d", "a", "d"],
            "weight": pd.Series([1, -1, 1, -1, 1], dtype=np.int8),
        },
    )


@pytest.fixture
def grn_graph() -> nx.DiGraph:
    dg = nx.DiGraph()
    dg.add_edges_from(
        [
            ("a", "b", {"weight": 1}),
            ("b", "c", {"weight": -1}),
            ("c", "d", {"weight": 1}),
            ("d", "a", {"weight": -1}),
            ("b", "d", {"weight": 1}),
        ]
    )
    return dg


@pytest.fixture
def grn_array() -> np.ndarray[tuple[int, int], np.dtype[np.int16]]:
    return np.array(
        [
            [0, 1, 0, 0],
            [0, 0, -1, 1],
            [0, 0, 0, 1],
            [-1, 0, 0, 0],
        ],
        dtype=np.int8,
    )  # ty: ignore[invalid-return-type]


@pytest.fixture
def grn_sparray() -> sparse.dok_array:
    arr = sparse.dok_array((4, 4), dtype=np.int16)
    arr[0, 1] = 1
    arr[1, 2] = -1
    arr[1, 3] = 1
    arr[2, 3] = 1
    arr[3, 0] = -1
    return arr


def test_create_from_df(grn_df):
    test_grn = GRN(grn_df)
    assert set(test_grn.index) == {"a", "b", "c", "d"}
    assert test_grn.sparray.nnz == 5
    expected_array = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, 1],
                [-1, 0, 0, 0],
            ],
            dtype=np.int8,
        ),
        index=pd.Index(["a", "b", "c", "d"]),
        columns=pd.Index(["a", "b", "c", "d"]),
    )
    np.testing.assert_equal(
        test_grn._array.todense(),
        expected_array.loc[list(test_grn.index), :][list(test_grn.index)].to_numpy(),
    )


def test_create_from_network(grn_graph):
    test_grn = GRN(grn_graph)
    assert set(test_grn.index) == {"a", "b", "c", "d"}
    assert test_grn.sparray.nnz == 5
    expected_array = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, 1],
                [-1, 0, 0, 0],
            ],
            dtype=np.int16,
        ),
        index=pd.Index(["a", "b", "c", "d"]),
        columns=pd.Index(["a", "b", "c", "d"]),
    )
    np.testing.assert_equal(
        test_grn._array.todense(),
        expected_array.loc[list(test_grn.index), :][list(test_grn.index)].to_numpy(),
    )


def test_create_from_array(grn_array):
    test_grn = GRN(grn_array)
    test_grn.index = ["a", "b", "c", "d"]
    assert set(test_grn.index) == {"a", "b", "c", "d"}
    assert test_grn.sparray.nnz == 5
    expected_array = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, 1],
                [-1, 0, 0, 0],
            ],
            dtype=np.int16,
        ),
        index=pd.Index(["a", "b", "c", "d"]),
        columns=pd.Index(["a", "b", "c", "d"]),
    )
    np.testing.assert_equal(
        test_grn._array.todense(),
        expected_array.loc[list(test_grn.index), :][list(test_grn.index)].to_numpy(),
    )


def test_create_from_sparray(grn_sparray):
    test_grn = GRN(grn_sparray)
    test_grn.index = ["a", "b", "c", "d"]
    assert set(test_grn.index) == {"a", "b", "c", "d"}
    assert test_grn.sparray.nnz == 5
    expected_array = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, 1],
                [-1, 0, 0, 0],
            ],
            dtype=np.int16,
        ),
        index=pd.Index(["a", "b", "c", "d"]),
        columns=pd.Index(["a", "b", "c", "d"]),
    )
    np.testing.assert_equal(
        test_grn._array.todense(),
        expected_array.loc[list(test_grn.index), :][list(test_grn.index)].to_numpy(),
    )


def test_extract_graph(grn_graph):
    test_grn = GRN(grn_graph)
    assert nx.is_isomorphic(
        test_grn.graph, grn_graph, edge_match=iso.numerical_edge_match("weight", 1)
    )


def test_extract_array(grn_array):
    test_grn = GRN(grn_array)
    np.testing.assert_equal(test_grn.array, grn_array)


def test_extract_sparray(grn_sparray):
    test_grn = GRN(grn_sparray)
    np.testing.assert_equal(test_grn.sparray.todense(), grn_sparray.todense())


def test_extract_df(grn_df):
    test_grn = GRN(grn_df, weight_type=np.int8)
    # Reordering since the values of the rows, not their order is important
    pd.testing.assert_frame_equal(
        test_grn.df.sort_values("source").sort_values("target").reset_index(drop=True),
        grn_df.sort_values("source").sort_values("target").reset_index(drop=True),
        check_like=True,
    )
