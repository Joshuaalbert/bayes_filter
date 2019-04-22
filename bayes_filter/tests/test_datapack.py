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
