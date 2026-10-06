import sys

import torch
import torch.utils.model_zoo

from scripts.benchmark_nyu import NYUInputs, ROOT, build_model, model_config, summary


def test_random_initialization_never_loads_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Random inference must not load or download weights')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    monkeypatch.setattr(torch.utils.model_zoo, 'load_url', forbidden)
    torch.set_num_threads(1)
    model = build_model(model_config(2023, ROOT / 'data/nyudepthv2_h5'))
    assert model.graph_prop.Graph_Prop_Block.knn.k == 16
    assert 'torch_geometric' not in sys.modules


def test_fixed_camera_buffers_and_gather_kernels_are_preserved():
    from model.graphcspn import Graph_Prop
    module = Graph_Prop()
    x, y = module.camera()
    assert torch.equal(module.x_3d, x.float()) and torch.equal(module.y_3d, y.float())
    assert 'x_3d' in dict(module.named_buffers()) and 'x_3d' not in module.state_dict()
    assert torch.equal(module.conv_sum.weight, torch.ones_like(module.conv_sum.weight))
    for layer in (module.guide_to_graph, module.depth_to_graph, module.graph_to_depth):
        assert not layer.weight.requires_grad
        assert (layer.weight == 1).sum() == layer.out_channels


def test_nyu_input_is_deterministic_and_has_500_points():
    dataset = NYUInputs(ROOT / 'data/nyudepthv2_h5')
    first, repeat = dataset[0], dataset[0]
    assert len(dataset) == 654
    assert first['rgb'].shape == (1, 3, 228, 304)
    assert first['dep'].shape == (1, 1, 228, 304)
    assert (first['dep'] > 0).sum() == 500
    assert torch.equal(first['dep'], repeat['dep'])


def test_summary_uses_all_measurements():
    result = summary([10, 20, 30])
    assert result['count'] == 3 and result['mean_ms'] == 20
    assert result['p95_ms'] == 29 and result['fps'] == 50


def test_knn_includes_self_and_selects_nearest_points():
    from model.graphcspn import dense_knn_matrix
    points = torch.tensor([0., 1., 10.]).reshape(1, 1, 3, 1)
    index = dense_knn_matrix(points, k=2)
    assert index[0, 0, 0].tolist() == [0, 1]
    assert index[0, 0, 2].tolist() == [2, 1]
