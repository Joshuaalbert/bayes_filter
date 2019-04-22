from .common_setup import *

from ..datapack import DataPack
from ..misc import make_example_datapack
import os

def test_datapack():
    datapack = make_example_datapack(4,2,1,["X"],clobber=True, name=os.path.join(TEST_FOLDER,'test_datapack_data.h5'))
    phase,axes = datapack.phase
    datapack.phase = phase+1.
    phasep1, axes = datapack.phase
    assert np.all(np.isclose(phasep1, phase+1.))

# def test_datapack():
#     datapack = DataPack('test.h5',readonly=False)
#     datapack.current_solset = 'sol000'
#     datapack.select(ant="RS*")
#     phase, axes = datapack.phase
#     print(axes)
#     datapack.phase = np.ones_like(phase)
#     print(datapack.phase)
#     # # datapack.add_solset('test')
#     # print(datapack.soltabs)
#     # # print(datapack.directions)
#     # datapack.add_soltab('foo', ant=['CS001HBA0'], dir=['patch_0'], freq=[0., 1.])
#     # print(datapack)
#     # datapack.delete_soltab('foo')