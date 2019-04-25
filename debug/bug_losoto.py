from losoto.h5parm import h5parm
import numpy as np

if __name__ == '__main__':
    filename = 'debug_losoto.h5'
    solset = 'testSolset'
    H = h5parm(filename, readonly=False)
    pols = ['xx','yy']
    dirs = ['a','b']
    ants = ['c','d']
    times = np.array([0.,1.])
    vals = np.ones((2,2,2,2))
    H.makeSolset(solsetName=solset, addTables=True)
    solset = H.getSolset(solset)
    solset.makeSoltab('testSoltab', axesNames=['pol', 'dir', 'ant', 'time'],
                                            axesVals=[pols, dirs, ants, times],
                                            vals=vals, weights=np.ones_like(vals),
                                            weightDtype='f64')
    soltab = solset.getSoltab('testSoltab')
    print(soltab)