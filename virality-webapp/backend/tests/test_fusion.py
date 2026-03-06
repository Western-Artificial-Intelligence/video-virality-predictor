import numpy as np

from app.fusion import fuse_vectors


def test_concat_fusion_appends_text_presence_mask():
    v = np.array([1.0, 2.0], dtype=np.float32)
    a = np.array([3.0], dtype=np.float32)
    t = np.array([4.0, 5.0], dtype=np.float32)

    out = fuse_vectors("concat", v, a, t, text_present=1, append_mask=True)

    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 1.0]


def test_sum_pool_fusion_matches_padding_logic():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = np.array([10.0], dtype=np.float32)
    t = np.array([100.0, 200.0], dtype=np.float32)

    out = fuse_vectors("sum_pool", v, a, t, text_present=0, append_mask=True)

    # padded arrays:
    # v=[1,2,3], a=[10,0,0], t=[100,200,0] => [111,202,3] + mask
    assert out.tolist() == [111.0, 202.0, 3.0, 0.0]


def test_max_pool_fusion_matches_padding_logic():
    v = np.array([1.0, 2.0], dtype=np.float32)
    a = np.array([10.0, -1.0], dtype=np.float32)
    t = np.array([0.5], dtype=np.float32)

    out = fuse_vectors("max_pool", v, a, t, text_present=1, append_mask=True)
    assert out.tolist() == [10.0, 2.0, 1.0]
