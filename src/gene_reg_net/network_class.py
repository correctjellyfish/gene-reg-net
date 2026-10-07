"""
Base classes for Gene Regulatory Networks
"""

import warnings
from collections.abc import Iterable
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

class GRN:
    """
    Represents a Gene Regulatory Network with edges that can be either -1 or 1

    Parameters
    ----------
    network : pd.DataFrame or nx.DiGraph or scipy Sparse Array
        The gene regulatory network to represent, can be a

        * DataFrame: A dataframe with 3 columns, 'source', 'target', and 'weight'.
          Each row represents a regulatory relationship from 'source', to 'target'.
          The 'weight' column should contain -1, 0, and 1. A value of -1 represents
          repression, a value of 0 represents no interaction (all relationships not
          explicity provided will be given this value), a value of 1 represents
          activation. Default column names can be overridden with the `source`, `target`,
          and `weight` parameters.
        * DiGraph: A directed graph with the nodes representing the genes in the
          network. Each node represents a gene, and edges represent a regulatory
          relationship. The weight of the edge indicates the type of the
          relationship (1 for activating, -1 for repressing, 0 for no relationship).
          The edge weight by default is taken to be the 'weight' edge attribute,
          but this can be overridden with the `weight` parameter.
        * NDArray: An array representing the regulatory relationships,
          should be a square array with entry i,j representing a regulatory relationship
          from j to i. The values must be only -1, 0, or 1 with -1 representing
          repression, 1 representing activation, and 0 representing no relationship.
        * Sparse Array: An array representing the regulatory relationships,
          should be a square array with entry i,j representing a regulatory relationship
          from j to i. The values must be only -1, 0, or 1 with -1 representing
          repression, 1 representing activation, and 0 representing no relationship.

    source : str,optional
       Optional string to specify the column of the network DataFrame to find the source
       genes for the regulatory relationships
    target : str,optional
       Optional string to specify the column of the network DataFrame to find the target
       genes for the regulatory relationships
    weight : str,optional
        Optional string to specify the column of the network DataFrame or the edge
        attribute of the network DiGraph to find the weight (or type) of the regulatory
        relationship.
    """

    def __init__(
        self,
        network: pd.DataFrame | nx.DiGraph | sparse.sparray | np.ndarray,
        *,
        source: str | None = None,
        target: str | None = None,
        weight: str | None = None,
    ):
        self.source = source
        self.target = target
        self.weight = weight
        # Create the _array and _index representing the network
        match network:
            case pd.DataFrame():
                self._index, array = self._df_init(network)
            case nx.DiGraph():
                self._index, array = self._graph_init(network)  # ty: ignore[invalid-argument-type]
            case sparse.sparray():
                self._index, array = self._sparray_init(network)
            case np.ndarray():
                self._index, array = self._nparray_init(network)
            case t:
                raise TypeError(
                    f"Expected a DataFrame, DiGraph, or sparse array, received {type(t)}"
                )
        self._array = array.tocsr()
        self._array.eliminate_zeros()
        # Check that the weights are all -1, 0, or 1
        vals = np.unique(self._array.data)
        if (
            (len(vals) > 2)
            or (len(vals) == 1 and (vals[0] != -1 and vals[0] != 1))
            or (len(vals) == 2 and (vals[0] != -1 or vals[1] != 1))
        ):
            raise ValueError(
                f"Weights should only be -1, 0, or 1 but weights includes incorrect values: {vals}"
            )
        # Create caches for various return types
        self._digraph: nx.DiGraph | None = None
        self._dataframe: pd.DataFrame | None = None
        self._nparray: (
            np.ndarray[tuple[int, int], np.dtype[np.int16]] | None
        ) = None

    def _reset_cache(self):
        self._digraph = None
        self._dataframe = None
        self._sparray = None
        self._nparray = None

    def _df_init(
        self,
        network: pd.DataFrame,
    ) -> tuple[pd.Index, sparse.dok_array]:
        # Get the genes in the regulatory network
        idx = pd.Index(
            set(network[self._source].unique())
            | set(network[self._target].unique())
        )
        array = sparse.dok_array((len(idx), len(idx)), dtype=np.int16)
        for _, (s, t, w) in network[
            [self.source, self.target, self.weight]
        ].iterrows():
            array[idx.get_loc(t), idx.get_loc(s)] = np.int16(w)
        return idx, array

    def _graph_init(
        self, network: nx.DiGraph
    ) -> tuple[pd.Index, sparse.dok_array]:
        idx = pd.Index(network.nodes)
        array = sparse.dok_array((len(idx), len(idx)), dtype=np.int16)
        for u, v, d in network.edges(data=True):
            array[idx.get_loc(v), idx.get_loc(u)] = np.int16(d[self.weight])
        return idx, array

    def _sparray_init(
        self, network: sparse.sparray
    ) -> tuple[pd.Index, sparse.dok_array]:
        # NOTE: The type ignores are due to how scipy creates its sparse arrays,
        # the shape and todok is available for all the implementations,
        # just not on the base class directly
        if network.shape[0] != network.shape[1]:  # ty: ignore[unresolved-attribute]
            raise ValueError("Network must be a square matrix")
        idx = pd.RangeIndex(network.shape[0])  # ty: ignore[unresolved-attribute]
        array = network.todok()  # ty: ignore[unresolved-attribute]
        return idx, array

    def _nparray_init(
        self, network: np.ndarray
    ) -> tuple[pd.Index, sparse.dok_array]:
        if network.shape[0] != network.shape[1]:
            raise ValueError("Network must be a square matrix")
        idx = pd.RangeIndex(network.shape[0])
        array = sparse.dok_array(network, dtype=np.int16)
        return idx, array

    @property
    def source(self):
        """
        The column of the long-form DataFrame used to specify
        the regulator in regulatory relationships (i.e. which
        gene is activating/repressing the gene in the 'target'
        column).
        """
        return self._source

    @source.setter
    def source(self, source: str | None):
        self._source = source if source is not None else "source"

    @property
    def target(self):
        """
        The column of the long-form DataFrame used to specify
        the regulatory target in regulatory relationships (i.e. which
        gene which is activated/repressed by the gene in the 'source'
        column).
        """
        return self._target

    @target.setter
    def target(self, target: str | None):
        self._target = target if target is not None else "target"

    @property
    def weight(self):
        """
        The column in the long-form DataFrame or edge attribute in the DiGraph
        which is used to specify the regulatory relationship direction.
        The column/attribute should only include -1, 0, and 1, with
        -1 indicating repression, 1 indicating activation, and 0
        indicating no regulation. This property is also used as the
        edge attribute for weight when returning a DiGraph from the
        `graph` property.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: str | None):
        self._weight = weight if weight is not None else "weight"

    @property
    def graph(self):
        """
        The gene regulatory network in the form of a DiGraph. Nodes represent
        genes in the network, and edges represent regulatory relationships.
        Each edge has an attribute (name specified by the `weight` property)
        specifying direction of regulation, -1 for repression, 1 for activation.
        """
        if self._digraph is not None:
            return self._digraph
        self._digraph = self._create_graph()
        return self._digraph

    @graph.setter
    def graph(self, network: nx.DiGraph, weight: str | None = None):
        self.weight = weight
        self._reset_cache()
        self._idx, array = self._graph_init(network=network)
        self._array: sparse.csr_array = array.tocsr()

    def _create_graph(self):
        assert isinstance(self._array, sparse.csr_array)
        coo = self._array.tocoo()
        row_idx, col_idx = coo.coords
        vals = coo.data
        idx = self.index
        g = nx.DiGraph()
        g.add_edges_from(
            (idx[j], idx[i], {"weight": v})
            for i, j, v in zip(row_idx, col_idx, vals)
        )
        return g

    @property
    def sparray(self):
        """
        The gene regulatory network in the form of a sparse array.
        Each i,j entry represents a regulatory relationship, with
        gene j regulating gene i. An entry of -1 represents
        gene j repressing gene i, 1 represents gene j activating
        gene i, and 0 indicates no regulatory relationship
        from gene j to gene i.
        """
        return self._array

    @sparray.setter
    def sparray(self, network: sparse.sparray):
        self._reset_cache()
        self._idx, array = self._sparray_init(network=network)
        self._array: sparse.csr_array = array.tocsr()

    @property
    def array(self):
        """
        The gene regulatory network in the form of a numpy array.
        Each i,j entry represents a regulatory relationship, with
        gene j regulating gene i. An entry of -1 represents
        gene j repressing gene i, 1 represents gene j activating
        gene i, and 0 indicates no regulatory relationship
        from gene j to gene i.
        """
        if self._nparray is not None:
            return self._nparray
        self._nparray = self._array.todense()
        return self._nparray

    @array.setter
    def array(self, network: np.ndarray[tuple[int, int], np.dtype[np.int16]]):
        self._reset_cache()
        self._idx, array = self._nparray_init(network=network)
        self._array: sparse.csr_array = array.tocsr()

    @property
    def df(self):
        """
        The gene regulatory network in the form of a pandas DataFrame.
        Row and column indexes are the gene labels.
        Each i,j entry represents a regulatory relationship, with
        gene j regulating gene i. An entry of -1 represents
        gene j repressing gene i, 1 represents gene j activating
        gene i, and 0 indicates no regulatory relationship
        from gene j to gene i.
        """
        if self._dataframe is not None:
            return self._dataframe
        self._dataframe = pd.DataFrame(self._array.todense(), index=self.index)
        return self._dataframe

    @df.setter
    def df(
        self,
        network: pd.DataFrame,
        source: str | None = None,
        target: str | None = None,
        weight: str | None = None,
    ):
        self._reset_cache()
        self.source = source
        self.target = target
        self.weight = weight
        self._idx, array = self._df_init(network=network)
        self._array: sparse.csr_array = array.tocsr()

    @property
    def index(self):
        """
        The labels for the genes in the regulatory network, in the
        order they appear in the underlying sparse array representation
        of the network (and thus also the arrays from the `sparray`
        and `array` properties). Used for naming nodes when for the
        `graph` property, and as the index and columns for the
        `df` property.
        """
        return self._index

    @index.setter
    def index(self, labels: Iterable[str]):
        self._reset_cache()
        idx = pd.Index(labels)
        if len(idx) != self._array.shape[0]:
            raise ValueError(
                f"Index must be the same length as the number of genes in the"
                f" regulatory network, but network has {self._array.shape[0]}"
                f" genes and there are {len(idx)} labels in the provided index"
            )
        self._index = idx

