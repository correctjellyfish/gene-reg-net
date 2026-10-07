"""
Base classes for Gene Regulatory Networks
"""

from collections.abc import Iterable
from typing import Generic, TypeVar

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

EdgeType = TypeVar("EdgeType", bound=np.dtype)


class GRN(Generic[EdgeType]):
    """
    Represents a Gene Regulatory Network

    Parameters
    ----------
    network : pd.DataFrame or nx.DiGraph or scipy Sparse Array
        The gene regulatory network to represent, can be a

        * DataFrame: A dataframe with 3 columns, 'source', 'target', and 'weight'.
          Each row represents a regulatory relationship from 'source', to 'target'.
          The 'weight' column should contain the regulatory weight between 'source'
          and 'target'. Negative values represent repression, positive values represent
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
          from i to j. The values must be only -1, 0, or 1 with -1 representing
          repression, 1 representing activation, and 0 representing no relationship.
        * Sparse Array: An array representing the regulatory relationships,
          should be a square array with entry i,j representing a regulatory relationship
          from i to j. The values must be only -1, 0, or 1 with -1 representing
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
                self._index, array, weight_type = self._df_init(network)
            case nx.DiGraph():
                self._index, array, weight_type = self._graph_init(network)
            case sparse.sparray():
                self._index, array, weight_type = self._sparray_init(network)
            case np.ndarray():
                self._index, array, weight_type = self._nparray_init(network)
            case t:
                raise TypeError(
                    f"Expected a DataFrame, DiGraph, or sparse array, received {type(t)}"
                )
        self._array = array.tocsr()
        self._array.eliminate_zeros()
        self._weight_type = weight_type
        # Create caches for various return types
        self._digraph: nx.DiGraph | None = None
        self._dataframe: pd.DataFrame | None = None
        self._nparray: np.ndarray[tuple[int, int], EdgeType] | None = None

    def _reset_cache(self):
        self._digraph = None
        self._dataframe = None
        self._sparray = None
        self._nparray = None

    def _df_init(
        self,
        network: pd.DataFrame,
    ) -> tuple[pd.Index, sparse.dok_array, np.dtype]:
        # Get the genes in the regulatory network
        idx = pd.Index(
            set(network[self._source].unique()) | set(network[self._target].unique())
        )
        if len(idx) == 0:
            return idx, sparse.dok_array((0, 0)), np.dtype(np.int_)
        array = sparse.dok_array((len(idx), len(idx)), dtype=network[self.weight].dtype)
        for _, (s, t, w) in network[[self.source, self.target, self.weight]].iterrows():
            array[idx.get_loc(s), idx.get_loc(t)] = w
        return idx, array, network[self.weight].dtype

    def _graph_init(
        self, network: nx.DiGraph
    ) -> tuple[pd.Index, sparse.dok_array, np.dtype]:
        idx = pd.Index(network.nodes)
        if len(idx) == 0:
            return idx, sparse.dok_array((0, 0)), np.dtype(np.int_)
        edges = network.edges(data=True)
        if len(edges == 0):
            return idx, sparse.dok_array((len(idx), len(idx))), np.dtype(np.int_)
        first_edge_data = edges.__iter__().__next__()[2]
        if self.weight not in first_edge_data:
            raise KeyError(
                f"Couldn't find expected key {self.weight} in the edge data of provided network, "
                f"try setting the 'weight' parameter to reflect which edge attribute holds the interaction weight"
            )
        weight_type = np.dtype(type(first_edge_data[self.weight]))
        array = sparse.dok_array((len(idx), len(idx)), dtype=weight_type)
        for u, v, d in network.edges(data=True):
            array[idx.get_loc(u), idx.get_loc(v)] = d[self.weight]
        return idx, array, weight_type

    def _sparray_init(
        self, network: sparse.sparray
    ) -> tuple[pd.Index, sparse.dok_array, np.dtype]:
        # NOTE: The type ignores are due to how scipy creates its sparse arrays,
        # the shape and todok is available for all the implementations,
        # just not on the base class directly
        if network.shape[0] != network.shape[1]:  # ty: ignore[unresolved-attribute]
            raise ValueError("Network must be a square matrix")
        idx = pd.RangeIndex(network.shape[0])  # ty: ignore[unresolved-attribute]
        array = network.todok()  # ty: ignore[unresolved-attribute]
        return idx, array, array.dtype

    def _nparray_init(
        self, network: np.ndarray
    ) -> tuple[pd.Index, sparse.dok_array, np.dtype]:
        if network.shape[0] != network.shape[1]:
            raise ValueError("Network must be a square matrix")
        idx = pd.RangeIndex(network.shape[0])
        array = sparse.dok_array(network, dtype=network.dtype)
        return idx, array, array.dtype

    @property
    def source(self):
        """
        The column of the long-form DataFrame used to specify
        the regulator in regulatory relationships (i.e. which
        gene is activating/repressing the gene in the 'target'
        column).
        """
        return self._source if self._source is not None else "source"

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
        return self._target if self._target is not None else "target"

    @target.setter
    def target(self, target: str | None):
        self._target = target if target is not None else "target"

    @property
    def weight(self):
        """
        The column in the long-form DataFrame or edge attribute in the DiGraph
        which is used to specify the regulatory relationship direction.
        This property is also used as the edge attribute for weight
        when returning a DiGraph from the `graph` property.
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
        specifying direction of regulation, negative for repression, positive
        for activation.
        """
        if self._digraph is not None:
            return self._digraph
        self._digraph = self._create_graph()
        return self._digraph

    @graph.setter
    def graph(self, network: nx.DiGraph, weight: str | None = None):
        self.weight = weight
        self._reset_cache()
        self._idx, array, weight_type = self._graph_init(network=network)
        self._weight_type = weight_type
        self._array: sparse.csr_array = array.tocsr()

    def _create_graph(self):
        assert isinstance(self._array, sparse.csr_array)
        coo = self._array.tocoo()
        row_idx, col_idx = coo.coords
        vals = coo.data
        idx = self.index
        g = nx.DiGraph()
        g.add_edges_from(
            (idx[i], idx[j], {"weight": v}) for i, j, v in zip(row_idx, col_idx, vals)
        )
        return g

    @property
    def sparray(self):
        """
        The gene regulatory network in the form of a sparse array.
        Each i,j entry represents a regulatory relationship, with
        gene i regulating gene j. A negative entry represents
        gene i repressing gene j, a positive entry represents gene i activating
        gene j, and 0 indicates no regulatory relationship
        from gene i to gene j.
        """
        return self._array

    @sparray.setter
    def sparray(self, network: sparse.sparray):
        self._reset_cache()
        self._idx, array, weight_type = self._sparray_init(network=network)
        self._weight_type = weight_type
        self._array: sparse.csr_array = array.tocsr()

    @property
    def array(self):
        """
        The gene regulatory network in the form of a numpy array.
        Each i,j entry represents a regulatory relationship, with
        gene i regulating gene j. A negative entry represents
        gene i repressing gene j, a positive entry represents
        gene i activating gene j, and 0 indicates no regulatory
        relationship from gene i to gene j.
        """
        if self._nparray is not None:
            return self._nparray
        self._nparray = self._array.todense()
        return self._nparray

    @array.setter
    def array(self, network: np.ndarray[tuple[int, int], EdgeType]):
        self._reset_cache()
        self._idx, array, weight_type = self._nparray_init(network=network)
        self._weight_type = weight_type
        self._array: sparse.csr_array = array.tocsr()

    @property
    def df(self):
        """
        The gene regulatory network in the form of a pandas DataFrame.
        Row and column indexes are the gene labels.
        Each i,j entry represents a regulatory relationship, with
        gene i regulating gene j. A negative entry represents
        gene i repressing gene j, a positive entry represents
        gene i activating gene j, and 0 indicates no regulatory
        relationship from gene i to gene j.
        """
        if self._dataframe is not None:
            return self._dataframe
        n_edges = self._array.nnz
        long_df = pd.DataFrame(
            {
                self._source: pd.Series(dtype=self._idx.dtype),
                self._target: pd.Series(dtype=self._idx.dtype),
                self._weight: pd.Series(dtype=self._weight_type),
            },
            index=pd.RangeIndex(n_edges),
        )
        coo = self._array.tocoo()
        row_idx, col_idx = coo.coords
        vals = coo.data
        for idx, (i, j, v) in enumerate(zip(row_idx, col_idx, vals)):
            long_df.loc[idx, "source"] = self.index[i]
            long_df.loc[idx, "target"] = self.index[j]
            long_df.loc[idx, "weight"] = v
        self._dataframe = long_df

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
        self._idx, array, weight_type = self._df_init(network=network)
        self._weight_type = weight_type
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
