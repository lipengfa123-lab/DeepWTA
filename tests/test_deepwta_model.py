from __future__ import annotations

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.deepwta_model_config import MODEL_CONFIG
from deepwta.models.deepwta_model import DeepWTAModel


def build_graph(num_bases: int, num_targets: int, seed: int) -> Data:
    g = torch.Generator().manual_seed(seed)
    k = MODEL_CONFIG.num_weapons
    l = MODEL_CONFIG.num_schemes

    # Node features.
    x = torch.zeros(num_bases + num_targets, MODEL_CONFIG.node_dim)
    x[:, :2] = torch.rand(num_bases + num_targets, 2, generator=g)
    x[num_bases:, 2] = 1.0

    # Base inventory B.
    x[:num_bases, 3:3 + k] = 0.25 + 0.70 * torch.rand(
        num_bases, k, generator=g
    )

    # Target scheme demand U.
    offset = 3 + k
    demand = 0.75 * torch.rand(num_targets, l, k, generator=g)
    demand[demand < 0.35] = 0.0
    x[num_bases:, offset:offset + l * k] = demand.reshape(num_targets, -1)

    # Target damage levels G.
    offset += l * k
    n1 = l // 3
    n2 = l // 3
    levels = torch.tensor(
        [1] * n1 + [2] * n2 + [3] * (l - n1 - n2),
        dtype=torch.float32,
    )
    x[num_bases:, offset:offset + l] = levels.unsqueeze(0) / 3.0

    # Target damage threshold G_th.
    thresholds = torch.tensor([1.0, 2.0, 3.0])
    x[num_bases:, -1] = thresholds[torch.arange(num_targets) % 3] / 3.0

    # Complete base -> target graph.
    src = torch.arange(num_bases).repeat_interleave(num_targets)
    dst = torch.arange(num_targets).repeat(num_bases) + num_bases
    edge_index = torch.stack([src, dst])

    num_edges = num_bases * num_targets
    edge_attr = torch.zeros(num_edges, MODEL_CONFIG.edge_dim)

    # D_norm.
    edge_attr[:, 0] = torch.rand(num_edges, generator=g)

    # Q.
    edge_attr[:, 1:1 + k] = 0.03 + 0.30 * torch.rand(
        num_edges, k, generator=g
    )

    # X.
    reachable = torch.ones(num_edges, k)
    reachable[0, 0] = 0.0
    if num_edges > 2 and k > 3:
        reachable[2, 3] = 0.0

    edge_attr[:, 1 + k:1 + 2 * k] = reachable

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def make_model(
    use_scheme_efficiency_features: bool = False,
    tactical_assignment_bias: str = "none",
    tactical_bias_alpha_init: float = 0.05,
    assignment_temperature: float = 1.0,
    zero_inventory_assignment_mask: bool = False,
    resource_feasible_scheme_mask: bool = False,
    positive_inventory_contention_lambda: float = 0.0,
    redistribution_mode: str = "none",
    redistribution_beta: float = 0.0,
    redistribution_iterations: int = 2,
    redistribution_eta: float = 0.75,
) -> DeepWTAModel:
    c = MODEL_CONFIG
    return DeepWTAModel(
        node_dim=c.node_dim,
        edge_dim=c.edge_dim,
        num_weapons=c.num_weapons,
        num_schemes=c.num_schemes,
        inventory_max=c.inventory_max,
        demand_max=c.demand_max,
        hidden_dim=c.hidden_dim,
        heads=c.heads,
        num_encoder_layers=c.num_encoder_layers,
        scheme_dim=c.scheme_dim,
        scheme_hidden_dim=c.scheme_hidden_dim,
        edge_mlp_dim=c.edge_mlp_dim,
        dropout=c.dropout,
        survival_floor=c.survival_floor,
        use_inventory_projection=c.use_inventory_projection,
        use_scheme_efficiency_features=use_scheme_efficiency_features,
        tactical_assignment_bias=tactical_assignment_bias,
        tactical_bias_alpha_init=tactical_bias_alpha_init,
        assignment_temperature=assignment_temperature,
        zero_inventory_assignment_mask=zero_inventory_assignment_mask,
        resource_feasible_scheme_mask=resource_feasible_scheme_mask,
        positive_inventory_contention_lambda=positive_inventory_contention_lambda,
        redistribution_mode=redistribution_mode,
        redistribution_beta=redistribution_beta,
        redistribution_iterations=redistribution_iterations,
        redistribution_eta=redistribution_eta,
    )


def test_single_graph():
    data = build_graph(4, 3, seed=1)

    # Narrow PyG optional attributes once, then use local tensors only.
    x = data.x
    edge_index = data.edge_index
    assert x is not None
    assert edge_index is not None

    model = make_model()
    model.train()

    out = model(data, scheme_mode="soft", scheme_tau=1.0)

    k = MODEL_CONFIG.num_weapons
    l = MODEL_CONFIG.num_schemes

    # Shapes.
    assert out.allocation.shape == (12, k)
    assert out.assignment_probs.shape == (12, k)
    assert out.scheme_probs.shape == (3, l)
    assert out.target_demand.shape == (3, k)
    assert out.route_load.shape == (12,)

    # X=0 -> allocation=0.
    assert out.allocation[0, 0].item() == 0.0

    # G < G_th -> scheme probability=0.
    offset = 3 + k + l * k
    target_x = x[x[:, 2] > 0.5]
    damage = target_x[:, offset:offset + l]
    threshold = target_x[:, -1:]
    infeasible = damage + 1e-6 < threshold
    assert torch.all(out.scheme_probs[infeasible] == 0.0)

    # Inventory: sum_j A[k,i,j] <= B[k,i].
    src = edge_index[0]
    used = torch.zeros(x.size(0), k)
    used.index_add_(0, src, out.allocation.detach())

    inventory = x[:, 3:3 + k] * MODEL_CONFIG.inventory_max
    assert torch.all(used <= inventory + 1e-4)

    # Continuous, finite outputs.
    assert torch.is_floating_point(out.allocation)
    assert torch.isfinite(out.allocation).all()
    assert torch.isfinite(out.scheme_probs).all()
    assert torch.isfinite(out.assignment_probs).all()


def test_batch_and_gradient():
    graph_a = build_graph(4, 3, seed=11)
    graph_b = build_graph(5, 2, seed=22)
    batch = Batch.from_data_list([graph_a, graph_b])

    model = make_model()
    model.train()

    out = model(batch, scheme_mode="soft", scheme_tau=0.8)

    k = MODEL_CONFIG.num_weapons
    l = MODEL_CONFIG.num_schemes
    expected_edges = 4 * 3 + 5 * 2
    expected_targets = 3 + 2

    assert out.allocation.shape == (expected_edges, k)
    assert out.scheme_probs.shape == (expected_targets, l)
    assert out.edge_target_index.shape == (expected_edges,)
    assert out.target_node_index.shape == (expected_targets,)
    assert out.edge_target_index.min().item() >= 0
    assert out.edge_target_index.max().item() < expected_targets

    loss = (
        out.allocation.mean()
        + out.scheme_probs.square().mean()
        + out.assignment_probs.square().mean()
    )
    loss.backward()

    grad_sum = sum(
        p.grad.abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    assert grad_sum > 0.0


@pytest.mark.parametrize(
    ("num_bases", "num_targets"),
    ((20, 10), (10, 20)),
)
def test_scheme_efficiency_features_variable_rectangular_graphs(
    num_bases: int,
    num_targets: int,
):
    graph = build_graph(num_bases, num_targets, seed=103)
    model = make_model(use_scheme_efficiency_features=True)
    model.train()

    out = model(graph, scheme_mode="soft", scheme_tau=1.0)
    features = out.scheme_efficiency_features
    assert features is not None
    assert features.shape == (
        num_targets,
        MODEL_CONFIG.num_schemes,
        5,
    )
    assert torch.isfinite(features).all()
    assert torch.all((features >= 0.0) & (features <= 1.0))
    assert torch.isfinite(out.allocation).all()

    out.scheme_probs.square().mean().backward()
    final_layer = model.scheme_efficiency_encoder[-1]
    assert final_layer.weight.grad is not None
    assert final_layer.weight.grad.abs().sum().item() > 0.0


def test_scheme_efficiency_features_asymmetric_batch_and_no_reachable_base():
    graphs = [
        build_graph(3, 7, seed=301),
        build_graph(8, 2, seed=302),
        build_graph(5, 4, seed=303),
    ]

    # Make target 0 in the first graph unreachable by every base/weapon.  The
    # efficiency statistics must use their finite fallback values in this case.
    first = graphs[0]
    k = MODEL_CONFIG.num_weapons
    unreachable_edges = first.edge_index[1] == 3
    first.edge_attr[unreachable_edges, 1 + k : 1 + 2 * k] = 0.0

    batch = Batch.from_data_list(graphs)
    model = make_model(use_scheme_efficiency_features=True).eval()
    with torch.no_grad():
        out = model(batch, scheme_mode="hard")

    features = out.scheme_efficiency_features
    assert features is not None
    assert features.shape == (
        sum(graph.num_nodes - int((graph.x[:, 2] < 0.5).sum()) for graph in graphs),
        MODEL_CONFIG.num_schemes,
        5,
    )
    assert torch.isfinite(features).all()
    assert torch.all((features >= 0.0) & (features <= 1.0))
    assert torch.isfinite(out.allocation).all()

    # With no reachable route, normalized reachability difficulty and the
    # conservative interception/distance fallbacks are exactly one.
    assert torch.allclose(features[0, :, 2:], torch.ones_like(features[0, :, 2:]))


def test_scheme_efficiency_old_checkpoint_compatibility_and_zero_init():
    graph = build_graph(5, 3, seed=211)
    baseline = make_model(use_scheme_efficiency_features=False).eval()
    enhanced = make_model(use_scheme_efficiency_features=True).eval()

    incompatible = enhanced.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith("scheme_efficiency_encoder.")
        for key in incompatible.missing_keys
    )

    with torch.no_grad():
        baseline_out = baseline(graph, scheme_mode="hard")
        enhanced_out = enhanced(graph, scheme_mode="hard")

    assert torch.equal(baseline_out.scheme_probs, enhanced_out.scheme_probs)
    assert torch.equal(baseline_out.allocation, enhanced_out.allocation)


@pytest.mark.parametrize(
    "bias_mode",
    ("distance", "interception", "both"),
)
def test_tactical_assignment_bias_formula_and_gradient(bias_mode: str):
    graph = build_graph(6, 4, seed=401)
    baseline = make_model().eval()
    biased = make_model(
        tactical_assignment_bias=bias_mode,
        tactical_bias_alpha_init=0.05,
    ).eval()
    incompatible = biased.load_state_dict(baseline.state_dict(), strict=False)
    expected_missing = set()
    if bias_mode in {"distance", "both"}:
        expected_missing.add("tactical_distance_raw_alpha")
    if bias_mode in {"interception", "both"}:
        expected_missing.add("tactical_interception_raw_alpha")
    assert set(incompatible.missing_keys) == expected_missing
    assert not incompatible.unexpected_keys

    biased.train()
    out = biased(graph, scheme_mode="soft", scheme_tau=1.0)
    assert out.assignment_net_logits is not None
    assert out.assignment_tactical_bias is not None
    expected = torch.zeros_like(out.assignment_tactical_bias)
    k = MODEL_CONFIG.num_weapons
    if bias_mode in {"distance", "both"}:
        alpha_d = torch.nn.functional.softplus(
            biased.tactical_distance_raw_alpha
        )
        assert alpha_d.item() >= 0.0
        expected -= alpha_d * graph.edge_attr[:, 0:1]
    if bias_mode in {"interception", "both"}:
        alpha_q = torch.nn.functional.softplus(
            biased.tactical_interception_raw_alpha
        )
        assert alpha_q.item() >= 0.0
        expected -= alpha_q * graph.edge_attr[:, 1 : 1 + k]

    assert torch.allclose(out.assignment_tactical_bias, expected)
    assert torch.allclose(
        out.assignment_logits,
        out.assignment_net_logits + out.assignment_tactical_bias,
    )
    assert torch.isfinite(out.assignment_tactical_bias).all()

    out.allocation.square().mean().backward()
    for name, parameter in biased.named_parameters():
        if name.startswith("tactical_"):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_tactical_assignment_bias_disabled_is_exact_baseline():
    graph = build_graph(5, 3, seed=402)
    baseline = make_model().eval()
    disabled = make_model(
        tactical_assignment_bias="none",
        tactical_bias_alpha_init=0.10,
    ).eval()
    disabled.load_state_dict(baseline.state_dict(), strict=True)
    with torch.no_grad():
        baseline_out = baseline(graph, scheme_mode="hard")
        disabled_out = disabled(graph, scheme_mode="hard")
    assert torch.equal(baseline_out.assignment_logits, disabled_out.assignment_logits)
    assert torch.equal(baseline_out.assignment_probs, disabled_out.assignment_probs)
    assert torch.equal(baseline_out.allocation, disabled_out.allocation)
    assert disabled_out.assignment_tactical_bias is None


def test_assignment_temperature_one_is_exact_baseline_and_sharpening_is_finite():
    graph = build_graph(7, 5, seed=501)
    baseline = make_model().eval()
    temperature_one = make_model(assignment_temperature=1.0).eval()
    sharper = make_model(assignment_temperature=0.6).eval()
    temperature_one.load_state_dict(baseline.state_dict(), strict=True)
    sharper.load_state_dict(baseline.state_dict(), strict=True)

    with torch.no_grad():
        baseline_out = baseline(graph, scheme_mode="hard")
        temperature_one_out = temperature_one(graph, scheme_mode="hard")
        sharper_out = sharper(graph, scheme_mode="hard")

    assert torch.equal(
        baseline_out.assignment_probs,
        temperature_one_out.assignment_probs,
    )
    assert torch.equal(baseline_out.allocation, temperature_one_out.allocation)
    assert torch.isfinite(sharper_out.assignment_probs).all()
    assert torch.isfinite(sharper_out.allocation).all()

    k = MODEL_CONFIG.num_weapons
    reachable = graph.edge_attr[:, 1 + k : 1 + 2 * k] > 0.5
    assert torch.all(sharper_out.assignment_probs[~reachable] == 0.0)
    baseline_entropy = -(
        baseline_out.assignment_probs
        * torch.log(baseline_out.assignment_probs.clamp_min(1e-12))
    ).sum()
    sharper_entropy = -(
        sharper_out.assignment_probs
        * torch.log(sharper_out.assignment_probs.clamp_min(1e-12))
    ).sum()
    assert sharper_entropy <= baseline_entropy + 1e-6


def test_zero_inventory_assignment_mask_disabled_is_exact_baseline():
    graph = build_graph(5, 3, seed=551)
    baseline = make_model().eval()
    explicitly_disabled = make_model(
        zero_inventory_assignment_mask=False
    ).eval()
    explicitly_disabled.load_state_dict(baseline.state_dict(), strict=True)

    with torch.no_grad():
        baseline_out = baseline(graph, scheme_mode="hard")
        disabled_out = explicitly_disabled(graph, scheme_mode="hard")

    assert torch.equal(baseline_out.assignment_probs, disabled_out.assignment_probs)
    assert torch.equal(
        baseline_out.allocation_pre_projection,
        disabled_out.allocation_pre_projection,
    )
    assert torch.equal(baseline_out.allocation, disabled_out.allocation)


def test_zero_inventory_assignment_mask_handles_all_masked_weapon_target():
    num_bases = 4
    num_targets = 3
    graph = build_graph(num_bases, num_targets, seed=552)
    k = MODEL_CONFIG.num_weapons
    original_reachability = graph.edge_attr[:, 1 + k : 1 + 2 * k].clone()

    # Weapon 0 has no stock at any base: every (weapon=0,target) group is
    # all-masked. Weapon 1 retains stock except at base 0, exercising the
    # ordinary partial mask in the same forward pass.
    graph.x[:num_bases, 3] = 0.0
    graph.x[0, 4] = 0.0
    model = make_model(zero_inventory_assignment_mask=True).eval()

    with torch.no_grad():
        output = model(graph, scheme_mode="hard")

    assert torch.isfinite(output.assignment_probs).all()
    assert torch.isfinite(output.allocation_pre_projection).all()
    assert torch.isfinite(output.allocation).all()
    assert torch.all(output.assignment_probs[:, 0] == 0.0)
    assert torch.all(output.allocation_pre_projection[:, 0] == 0.0)
    assert torch.all(output.allocation[:, 0] == 0.0)

    src = graph.edge_index[0]
    target = graph.edge_index[1] - num_bases
    assert torch.all(output.assignment_probs[src == 0, 1] == 0.0)
    probability_sums = torch.zeros(num_targets)
    probability_sums.index_add_(0, target, output.assignment_probs[:, 1])
    assert torch.allclose(probability_sums, torch.ones_like(probability_sums))
    assert torch.equal(
        graph.edge_attr[:, 1 + k : 1 + 2 * k],
        original_reachability,
    )


def test_one_step_contention_correction_matches_closed_form_update():
    num_bases = 5
    num_targets = 4
    graph = build_graph(num_bases, num_targets, seed=561)
    k = MODEL_CONFIG.num_weapons
    # Small, heterogeneous positive inventories force nonuniform overload while
    # preserving the positive-stock mask for every weapon.
    graph.x[:num_bases, 3 : 3 + k] = torch.linspace(
        0.001, 0.010, steps=num_bases * k
    ).reshape(num_bases, k)
    c2 = make_model(zero_inventory_assignment_mask=True).eval()
    corrected = make_model(
        zero_inventory_assignment_mask=True,
        positive_inventory_contention_lambda=0.5,
    ).eval()
    corrected.load_state_dict(c2.state_dict(), strict=True)

    with torch.no_grad():
        first = c2(graph, scheme_mode="hard")
        output = corrected(graph, scheme_mode="hard")

    src, dst = graph.edge_index
    edge_q = graph.edge_attr[:, 1 : 1 + k]
    edge_x = graph.edge_attr[:, 1 + k : 1 + 2 * k]
    inventory_norm = graph.x[:, 3 : 3 + k]
    assignment_mask = edge_x * (inventory_norm[src] > 0.0).to(edge_x.dtype)
    survival = (1.0 - edge_q).clamp_min(corrected.survival_floor)
    demand_on_edge = first.target_demand[first.edge_target_index]
    allocation_first = first.assignment_probs * demand_on_edge / survival * assignment_mask
    load = torch.zeros(graph.num_nodes, k)
    load.index_add_(0, src, allocation_first)
    inventory = inventory_norm * MODEL_CONFIG.inventory_max
    rho = torch.where(
        inventory > 0.0,
        load / inventory.clamp_min(1e-8),
        torch.zeros_like(load),
    )
    pressure = torch.log1p(torch.relu(rho - 1.0))
    expected_logits = output.assignment_logits - 0.5 * pressure[src]
    expected = softmax(
        expected_logits.masked_fill(assignment_mask <= 0.5, -1.0e4),
        dst,
        dim=0,
    ) * assignment_mask
    sums = torch.zeros(graph.num_nodes, k)
    sums.index_add_(0, dst, expected)
    expected = torch.where(
        sums[dst] > 1e-8,
        expected / sums[dst].clamp_min(1e-8),
        torch.zeros_like(expected),
    )

    assert torch.isfinite(output.assignment_probs).all()
    assert torch.allclose(output.assignment_probs, expected, atol=1e-6, rtol=1e-6)
    assert torch.max(torch.abs(output.assignment_probs - first.assignment_probs)) > 1e-6


def test_resource_feasible_scheme_mask_disabled_is_exact_baseline():
    graph = build_graph(5, 3, seed=571)
    baseline = make_model().eval()
    explicitly_disabled = make_model(
        resource_feasible_scheme_mask=False
    ).eval()
    explicitly_disabled.load_state_dict(baseline.state_dict(), strict=True)

    with torch.no_grad():
        baseline_out = baseline(graph, scheme_mode="hard")
        disabled_out = explicitly_disabled(graph, scheme_mode="hard")

    assert torch.equal(baseline_out.scheme_probs, disabled_out.scheme_probs)
    assert torch.equal(baseline_out.assignment_probs, disabled_out.assignment_probs)
    assert torch.equal(baseline_out.allocation, disabled_out.allocation)


def test_resource_feasible_scheme_mask_excludes_impossible_candidate():
    num_bases = 4
    num_targets = 2
    graph = build_graph(num_bases, num_targets, seed=572)
    k = MODEL_CONFIG.num_weapons
    l = MODEL_CONFIG.num_schemes
    demand_offset = 3 + k
    demand = graph.x[
        num_bases:, demand_offset : demand_offset + l * k
    ].view(num_targets, l, k)

    # Weapon 0 has no stock anywhere. For target 0 only scheme 5 avoids that
    # weapon, so the joint mask must select scheme 5 regardless of its logit.
    graph.x[:num_bases, 3] = 0.0
    demand[0, :, 0] = 0.5
    demand[0, 5, 0] = 0.0
    model = make_model(resource_feasible_scheme_mask=True).eval()

    with torch.no_grad():
        output = model(graph, scheme_mode="hard")

    assert torch.isfinite(output.scheme_probs).all()
    assert output.scheme_probs[0, 5] == 1.0
    assert output.scheme_probs[0].sum() == 1.0


def test_resource_feasible_scheme_mask_explicitly_falls_back_when_all_masked():
    num_bases = 4
    num_targets = 2
    graph = build_graph(num_bases, num_targets, seed=573)
    k = MODEL_CONFIG.num_weapons
    l = MODEL_CONFIG.num_schemes
    demand_offset = 3 + k
    demand = graph.x[
        num_bases:, demand_offset : demand_offset + l * k
    ].view(num_targets, l, k)
    graph.x[:num_bases, 3 : 3 + k] = 0.0
    demand[0] = 0.5  # no resource-feasible candidate for target 0

    baseline = make_model().eval()
    masked = make_model(resource_feasible_scheme_mask=True).eval()
    masked.load_state_dict(baseline.state_dict(), strict=True)
    with torch.no_grad():
        baseline_output = baseline(graph, scheme_mode="hard")
        masked_output = masked(graph, scheme_mode="hard")

    assert torch.isfinite(masked_output.scheme_probs).all()
    assert torch.equal(
        masked_output.scheme_probs[0],
        baseline_output.scheme_probs[0],
    )
    assert masked_output.scheme_probs[0].sum() == 1.0


@pytest.mark.parametrize(
    ("mode", "beta"),
    (("capacity", 0.0), ("network", 0.5), ("network", 1.0)),
)
def test_iterative_redistribution_is_finite_inventory_feasible_and_differentiable(
    mode: str,
    beta: float,
):
    graph = build_graph(7, 5, seed=601)
    model = make_model(
        redistribution_mode=mode,
        redistribution_beta=beta,
        redistribution_iterations=2,
        redistribution_eta=0.75,
    ).train()
    out = model(
        graph,
        scheme_mode="soft",
        scheme_tau=1.0,
        collect_redistribution_diagnostics=True,
    )
    diagnostics = out.redistribution_diagnostics
    assert diagnostics is not None
    assert diagnostics["normalized_shortage"].shape == (3,)
    assert diagnostics["shortage_pairs"].shape == (3,)
    assert diagnostics["shortage_targets"].shape == (3,)
    assert diagnostics["redistributed_shots_per_iteration"].shape == (2,)
    assert torch.isfinite(out.allocation).all()
    assert all(torch.isfinite(value).all() for value in diagnostics.values())

    k = MODEL_CONFIG.num_weapons
    src = graph.edge_index[0]
    used = torch.zeros(graph.num_nodes, k)
    used.index_add_(0, src, out.allocation.detach())
    inventory = graph.x[:, 3 : 3 + k] * MODEL_CONFIG.inventory_max
    assert torch.all(used <= inventory + 1e-4)

    out.allocation.square().mean().backward()
    grad_sum = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.assignment_decoder.parameters()
        if parameter.grad is not None
    )
    assert grad_sum > 0.0


def test_redistribution_disabled_is_exact_baseline():
    graph = build_graph(6, 4, seed=602)
    baseline = make_model().eval()
    disabled = make_model(
        redistribution_mode="none",
        redistribution_iterations=3,
        redistribution_eta=0.5,
    ).eval()
    disabled.load_state_dict(baseline.state_dict(), strict=True)
    with torch.no_grad():
        baseline_out = baseline(graph, scheme_mode="hard")
        disabled_out = disabled(graph, scheme_mode="hard")
    assert torch.equal(baseline_out.allocation, disabled_out.allocation)
    assert disabled_out.redistribution_diagnostics is None


if __name__ == "__main__":
    print("PyTorch:", torch.__version__)
    test_single_graph()
    test_batch_and_gradient()
    print("PASS: DeepWTAModel smoke tests")
