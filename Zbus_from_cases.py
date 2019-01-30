import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import networkx as nx


def kron_reduction(Matrix, IndPiv):
    """
        Returns new scipy.sparse.lil_matrix appliying kron's reduction to the row
        and column of the pivot element (IndPiv). It usses the Gabriel kron's
        reduction explained in Cap.7 Stevenson.

        param Matrix: Original squared matrix.
        type Matrix: sparse scipy matrix

        param IndPiv: Index of the pivot element within Matrix (0-indexed).
        type IndPiv: int

        Syntax:
            M_new = kron_reduction(M_prev, IndPiv)

        Example:
            >>> from scipy import sparse
            >>> kron_reduction( sparse.lil_matrix([[1, 2], [3, 4]]) , 1 ).todense()
            [[-0.5+0.j]]
    """
    Dim = Matrix.shape[0]
    # Assures lil_matrix format
    Matrix = Matrix.tolil()
    m_pp = Matrix[IndPiv, IndPiv]
    M = sparse.lil_matrix( (Dim - 1, Dim - 1), dtype=np.complex128 )
    RowsWoPiv = set(range(0, Dim)) - {IndPiv}  # index rows without pivot
    NewMFil = 0  # keeps the returned index matrix
    for row in RowsWoPiv:
        ColsWoPiv = list( set(range(0, Dim)) - {IndPiv})  # column index without pivot
        m_ij = Matrix[row, ColsWoPiv]
        m_ip = Matrix[row, IndPiv]
        m_pj = Matrix[IndPiv, ColsWoPiv]
        M[NewMFil, :] = m_ij - m_pj.multiply(m_ip) / m_pp
        NewMFil += 1
    return M


def make_Zbus(Graph):
    """
        Creates a Zbus from the input weighted Graph following the 4 cases for
        original Zbus modifications, described in Chapther 8 Stevenson's book
        (Table 8.1 sumarizes all four cases). Node '-1' is considered as reference,
        and must exists within Graph's nodes.

        Syntax:
            Zbus, NodeList = make_Zbus(Graph)

        :param Graph:
        :type Graph: networkx simple Graph

        Returns a tuple (Zbus, NodeList), where
            Zbus: scipy.sparse.coo_matrix() with constructed impedance matrix.
            NodeList: list of the index of nodes from Graph representig rows/columns of Zbus.

    """
    RefNode = -1

    def case1(Z_old, Zb):
        """
            Add Zb connected from new bus 'p' to reference.
            Adds a row and a column at the end of Z_old.
        """
        Dim = Z_old.shape[0]
        NewRow = sparse.coo_matrix((1, Dim), dtype=np.complex128)
        NewCol = sparse.coo_matrix((Dim, 1), dtype=np.complex128)
        Z_new = sparse.hstack([Z_old, NewCol])
        Z_new = sparse.vstack([Z_new, sparse.hstack([NewRow, Zb])])
        return Z_new

    def case2(Z_old, Zb, NodeIndxMat):
        """
            Add Zb connected from new bus 'p' to existing 'k' bus.
            Adds a row and a column at the end of Z_old.
        """
        NewRow = Z_old.getrow(NodeIndxMat)
        NewCol = Z_old.getcol(NodeIndxMat)
        Zkk = Z_old.tolil()[NodeIndxMat, NodeIndxMat]
        print("Zkk:", Zkk)
        Z_new = sparse.hstack([Z_old, NewCol])
        Z_new = sparse.vstack([Z_new, sparse.hstack([NewRow, Zkk + Zb])])
        return Z_new

    def case3(Z_old, Zb, NodeIndxMat):
        """
            Add Zb connected from existing bus 'k' to reference (temporal shorted 'p' bus).
            Adds a row and a column at the end of Z_old. Afterwords makes kron reduction.
        """
        NewRow = Z_old.getrow(NodeIndxMat)
        NewCol = Z_old.getcol(NodeIndxMat)
        Zkk = Z_old.tolil()[NodeIndxMat, NodeIndxMat]
        print("Zkk:", Zkk)
        Z_new = sparse.hstack([Z_old, NewCol])
        Z_new = sparse.vstack([Z_new, sparse.hstack([NewRow, Zkk + Zb])])
        # Apply kron reduction to added col & row
        Z_new = kron_reduction(Z_new, Z_new.shape[0] - 1)
        return Z_new

    def case4(Z_old, Zb, NodeLIndxMat, NodeKIndxMat):
        """
            Add Zb connected from existing bus 'k' to existing 'l' bus (temporal 'q' bus).
            Adds a row and a column at the end of Z_old, as the difference of l - k.
            Afterwords makes kron reduction. Bus 'k' connects to 'l' through Zb.
        """
        Z_old_lil = Z_old.tolil()
        NewRow = Z_old.getrow(NodeLIndxMat) - Z_old.getrow(NodeKIndxMat)
        NewCol = Z_old.getcol(NodeLIndxMat) - Z_old.getcol(NodeKIndxMat)
        Zkk = Z_old_lil[NodeLIndxMat, NodeLIndxMat]
        Zll = Z_old_lil[NodeKIndxMat, NodeKIndxMat]
        Zlk = Z_old_lil[NodeLIndxMat, NodeKIndxMat]
        Z_new = sparse.hstack([Z_old, NewCol])
        Z_new = sparse.vstack([Z_new, sparse.hstack([NewRow, Zkk + Zll - 2 * Zlk + Zb])])
        # Apply kron reduction to added col & row
        Z_new = kron_reduction(Z_new, Z_new.shape[0] - 1)
        return Z_new

    # Flag for first item
    First = True
    # inicialize ordered nodes already existing (without reference)
    NodesUsed = []
    for edge in Graph.edges(data=True):
        BusA, BusB, w = edge[0], edge[1], edge[2]['weight']
        print(BusA, BusB, w)
        if First:
            # initializa Zbus with First element
            Zbus = sparse.coo_matrix( w, dtype=np.complex128 )
            if (BusA not in NodesUsed) & (BusA != RefNode):
                NodesUsed.append(BusA)
            if (BusB not in NodesUsed) & (BusB != RefNode):
                NodesUsed.append(BusB)
            First = False
            print("Caso1")
            print("NodesUsed:", NodesUsed)
            print(Zbus.todense())
            print()
            continue
        #
        # Detect conditions for case selection
        # Case1
        CondCase1_Aref = (BusA == RefNode) & (BusB not in NodesUsed)
        CondCase1_Bref = ( (BusA not in NodesUsed) & (BusB == RefNode) )
        # Case2
        CondCase2_Anew = (BusA not in NodesUsed) & (BusA != RefNode) & (BusB in NodesUsed)
        CondCase2_Bnew = ( (BusA in NodesUsed) & (BusB not in NodesUsed) & (BusA != RefNode) )
        # Case3
        CondCase3_Aref = (BusA == RefNode) & (BusB in NodesUsed)
        CondCase3_Bref = ( (BusA in NodesUsed) & (BusB == RefNode) )
        # Case4
        CondCase4 = (BusA in NodesUsed) & (BusB in NodesUsed)
        #
        if CondCase1_Aref | CondCase1_Bref:
            print("Caso1")
            Zbus = case1(Zbus, w)
        elif CondCase2_Anew | CondCase2_Bnew:
            print("Caso2")
            if CondCase2_Anew:
                NodeIndxMat = NodesUsed.index(BusB)
            elif CondCase2_Bnew:
                NodeIndxMat = NodesUsed.index(BusA)
            Zbus = case2(Zbus, w, NodeIndxMat)
        elif CondCase3_Aref | CondCase3_Bref:
            print("Caso3")
            if CondCase3_Aref:
                NodeIndxMat = NodesUsed.index(BusB)
            elif CondCase3_Bref:
                NodeIndxMat = NodesUsed.index(BusA)
            Zbus = case3(Zbus, w, NodeIndxMat)
        elif CondCase4:
            print("Caso4")
            NodeLIndxMat = NodesUsed.index(BusA)
            NodeKIndxMat = NodesUsed.index(BusB)
            Zbus = case4(Zbus, w, NodeLIndxMat, NodeKIndxMat)
        else:
            print("Non edge is added")

        if (BusA not in NodesUsed) and (BusA != RefNode):
            NodesUsed.append(BusA)
        if (BusB not in NodesUsed) and (BusB != RefNode):
            NodesUsed.append(BusB)
        print("NodesUsed:", NodesUsed)
        print(Zbus.todense())
        print()
    print("Zbus complete!")
    return (Zbus, NodesUsed)


def ReorderSparseMatrix(SparseMatrix, OrderList):
    """
        Order the rows and columns (independently) of the squared SparseMatrix according to the order
        list OrderList. This function has the main purpose to ease the visualization of the results.
        Note if OrderList is [0,1,2,...,SparseMatrix.shape[0]] nothing changes.

        Syntax:
            >>> Matrix = ReorderSparseMatrix(Matrix, [0, 1, 3, 2, 4, ..., Matrix.shape[0]])
            >>> Matrix.todense()  # coo_matrix to dense
            [[...], [...]]  # matrix with columns swap according to OrderList, as well as rows.

        :param SparseMatrix: Source sparse matrix to be reordered. Must be Squared.
        :type SparseMatrix: Any type of scipy.sparse matrix

        :param OrderList: New order of the index matrix. Must have same length of SparseMatrix.
        :type OrderList: list
    """
    # Converts SparseMatrix to coo_matrix
    SparseMatrix = SparseMatrix.tocoo()  # assures coo_matrix type
    # Get the index of OrderList that makes it ascending sorted
    OrderedList = np.argsort(OrderList, kind='quicksort')
    # reasigns the rows and columns
    SparseMatrix.row = OrderedList[SparseMatrix.row]
    SparseMatrix.col = OrderedList[SparseMatrix.col]
    return SparseMatrix


# Example 8.4 (-1 is RefNode)
G = nx.Graph()
G.add_nodes_from([-1, 1, 5, 3, 4])
G.add_edge(-1, 1, weight = 1.25j)
G.add_edge(1, 5, weight = 0.25j)
G.add_edge(5, 3, weight = 0.4j)
G.add_edge(5, 4, weight = 0.125j)
G.add_edge(3, -1, weight = 1.25j)
G.add_edge(3, 4, weight = 0.2j)
print(G.nodes)
print(G.edges(data=True))
print()

Zbus, Nodes = make_Zbus(G)
print("Zbus:\n", Zbus.todense())
print("Nodes", Nodes)
print()

Zbus = ReorderSparseMatrix(Zbus, [0, 2, 1, 3])
# Zbus = ReorderSparseMatrix(Zbus, Nodes)
print("Zbus:\n", Zbus.todense())
Zbus = Zbus.tocsc(copy=False)

# proof needed for Ybus!
Ybus = linalg.inv(Zbus)
print("Ybus:\n", Ybus.todense())
