import cymorph.metrics as cm
import numpy as np
import unittest

class CyMorphTest(unittest.TestCase):
    def symmetricalCase(self):
        n = 2 + 2**(int(11*np.random.rand())+1)
        mat = np.random.rand(n,n)
        for i in range(n//2):
            for j in range(n//2):
                mat[(n-1)-(i+n//2)][(n-1)-(j+n//2)] = mat[i][j] 
                mat[i][(n-1)-(j+n//2)] = mat[i][j] 
                mat[(n-1)-(i+n//2)][j] = mat[i][j]
        self.assertEqual(cm.a2(mat), 1)
        self.assertEqual(cm.a3(mat), 1)

if __name__ == '__main__':
    unittest.main()
