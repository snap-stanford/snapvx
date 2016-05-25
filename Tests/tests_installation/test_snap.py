import sys
sys.path.append('..')

# Base class for unit tests.
from base_test import BaseTest
import unittest

class SnapTest(BaseTest):

    def test_snap(self):
        """Test that snap.py installed correctly.
        """
        import snap
        num_nodes = 20

        # Generate different undirected graphs
        full_graph = snap.GenFull(snap.PUNGraph, num_nodes)
        star_graph = snap.GenStar(snap.PUNGraph, num_nodes)
        random_graph = snap.GenRndGnm(snap.PUNGraph, num_nodes, num_nodes * 3)

        # Basic statistics on the graphs
        self.assertEqual(snap.CntInDegNodes(full_graph, num_nodes - 1), num_nodes)
        self.assertEqual(snap.CntOutDegNodes(full_graph, num_nodes - 1), num_nodes)
        self.assertEqual(snap.GetMxInDegNId(star_graph), 0)
        self.assertEqual(snap.GetMxOutDegNId(star_graph), 0)

        # Iterator
        degree_to_count = snap.TIntPrV()
        snap.GetInDegCnt(full_graph, degree_to_count)
        # There should be only one entry (num_nodes - 1, num_nodes)
        for item in degree_to_count:
            self.assertEqual(num_nodes - 1, item.GetVal1())
            self.assertEqual(num_nodes, item.GetVal2())

        # Rewiring
        rewired_graph = snap.GenRewire(random_graph)
        for n1 in random_graph.Nodes():
            for n2 in rewired_graph.Nodes():
                if n1.GetId() == n2.GetId():
                    self.assertEqual(n1.GetOutDeg() + n1.GetInDeg(),
                                     n2.GetOutDeg() + n2.GetInDeg())

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(SnapTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
