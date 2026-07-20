from __future__ import annotations

"""Structural node2vec embeddings over the solved-case dataset.

This is deliberately independent of the FaissCaseRetriever's text
embeddings (calcmate.retrieval): instead of embedding problem *text*, it
embeds each case's position in a similarity graph built from *structural*
attributes - which equations/constraints it uses, its target symbol,
concept, and difficulty. Two cases end up close in embedding space when
they tend to co-occur in each other's random-walk neighborhoods, i.e. when
they belong to the same structural neighborhood of the case graph, not
because their wording is similar.

Node2vec is transductive: it only produces vectors for nodes that were in
the graph at training time, so it can't directly embed a brand-new query.
The intended use (see calcmate.fallback_solver) is to anchor on a seed case
found by the existing text/structural retrievers, then use this module to
pull in further cases from the seed's structural neighborhood - a graph
expansion step layered on top of retrieval, not a replacement for it. A
future GNN (inductive) is the natural next step once this stops being
enough.
"""

import json
import math
import random
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_PATHS = [
    PROJECT_ROOT / "data" / "cases" / "kinematics_cases.jsonl",
    PROJECT_ROOT / "data" / "cases" / "kinematics_suvat_cases.jsonl",
]
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "cases" / "node2vec_embeddings.json"

# Similarity weights: how much each shared structural attribute contributes
# to the edge weight between two cases. Equations/constraints matter most
# since they determine whether one case's method actually transfers to
# another problem; concept/difficulty are weaker thematic signals.
_WEIGHTS = {
    "equations": 3.0,
    "constraints": 2.0,
    "unknown": 1.5,
    "concept": 1.0,
    "difficulty": 0.5,
    "known_symbols": 1.0,
}
_EDGE_THRESHOLD = 1.0


def load_all_cases(paths: list[Path] | None = None) -> list[dict]:
    """Load and de-duplicate (by case_id) every case across the dataset files."""
    cases: dict[str, dict] = {}
    for path in paths or DEFAULT_CASE_PATHS:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.setdefault(data["case_id"], data)
    return list(cases.values())


def _known_symbol_set(case: dict) -> set[str]:
    # Older records store "symbol=value" strings; newer ones store plain
    # symbols. Normalize to the bare symbol so Jaccard overlap is meaningful
    # across both.
    symbols = set()
    for raw in case.get("known_symbols", []) or []:
        symbols.add(str(raw).split("=")[0].strip())
    return symbols


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _case_similarity(a: dict, b: dict) -> float:
    score = 0.0
    eq_a, eq_b = set(a.get("equations_used") or []), set(b.get("equations_used") or [])
    score += _WEIGHTS["equations"] * _jaccard(eq_a, eq_b)

    con_a, con_b = set(a.get("constraints_fired") or []), set(b.get("constraints_fired") or [])
    score += _WEIGHTS["constraints"] * _jaccard(con_a, con_b)

    if a.get("unknown") and a.get("unknown") == b.get("unknown"):
        score += _WEIGHTS["unknown"]

    if a.get("concept") and a.get("concept") == b.get("concept"):
        score += _WEIGHTS["concept"]

    if a.get("difficulty") and a.get("difficulty") == b.get("difficulty"):
        score += _WEIGHTS["difficulty"]

    score += _WEIGHTS["known_symbols"] * _jaccard(_known_symbol_set(a), _known_symbol_set(b))
    return score


def build_case_graph(cases: list[dict]) -> nx.Graph:
    """Build a weighted similarity graph over cases.

    Each pair of cases sharing enough structure (same equations/constraints/
    target/concept) gets an edge; the weight controls how strongly node2vec's
    random walk favors that transition.
    """
    graph = nx.Graph()
    for case in cases:
        graph.add_node(case["case_id"], **{k: case.get(k) for k in ("concept", "difficulty", "unknown", "problem_text")})

    for i, case_a in enumerate(cases):
        for case_b in cases[i + 1 :]:
            weight = _case_similarity(case_a, case_b)
            if weight >= _EDGE_THRESHOLD:
                graph.add_edge(case_a["case_id"], case_b["case_id"], weight=weight)

    return graph


def _biased_neighbor(
    graph: nx.Graph, previous: str | None, current: str, p: float, q: float
) -> str | None:
    """Sample the next node in a 2nd-order random walk (the node2vec bias).

    p controls the likelihood of immediately revisiting `previous` (return
    parameter); q controls how far the walk explores away from `previous`'s
    neighborhood (in-out parameter). p=q=1 degenerates to an unbiased
    weighted walk (~DeepWalk).
    """
    neighbors = list(graph.neighbors(current))
    if not neighbors:
        return None

    previous_neighbors = set(graph.neighbors(previous)) if previous is not None else set()
    weights = []
    for neighbor in neighbors:
        base = graph[current][neighbor].get("weight", 1.0)
        if neighbor == previous:
            factor = 1.0 / p
        elif neighbor in previous_neighbors:
            factor = 1.0
        else:
            factor = 1.0 / q
        weights.append(base * factor)

    return random.choices(neighbors, weights=weights, k=1)[0]


def generate_walks(
    graph: nx.Graph, num_walks: int = 10, walk_length: int = 20, p: float = 1.0, q: float = 1.0
) -> list[list[str]]:
    nodes = list(graph.nodes)
    walks: list[list[str]] = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for start in nodes:
            walk = [start]
            previous: str | None = None
            current = start
            for _ in range(walk_length - 1):
                nxt = _biased_neighbor(graph, previous, current, p, q)
                if nxt is None:
                    break
                walk.append(nxt)
                previous, current = current, nxt
            walks.append(walk)
    return walks


def train_skipgram(
    walks: list[list[str]],
    dim: int = 32,
    window: int = 5,
    epochs: int = 5,
    negative_samples: int = 5,
    learning_rate: float = 0.02,
    seed: int = 20260717,
) -> dict[str, list[float]]:
    """Train node2vec's skip-gram-with-negative-sampling objective on the
    random walks using PyTorch. Equivalent in spirit to node2vec's usual
    gensim.Word2Vec backend, implemented directly so it doesn't depend on a
    package that requires a C build toolchain to install.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    random.seed(seed)

    vocabulary = sorted({node for walk in walks for node in walk})
    index_by_node = {node: i for i, node in enumerate(vocabulary)}
    vocab_size = len(vocabulary)
    if vocab_size == 0:
        return {}

    # Unigram^0.75 negative-sampling distribution (standard word2vec trick).
    frequency = [0] * vocab_size
    for walk in walks:
        for node in walk:
            frequency[index_by_node[node]] += 1
    freq_tensor = torch.tensor(frequency, dtype=torch.float32).clamp(min=1.0)
    neg_distribution = freq_tensor.pow(0.75)
    neg_distribution /= neg_distribution.sum()

    target_embed = nn.Embedding(vocab_size, dim)
    context_embed = nn.Embedding(vocab_size, dim)
    nn.init.uniform_(target_embed.weight, -0.5 / dim, 0.5 / dim)
    nn.init.zeros_(context_embed.weight)

    pairs: list[tuple[int, int]] = []
    for walk in walks:
        indices = [index_by_node[node] for node in walk]
        for center_pos, center in enumerate(indices):
            start = max(0, center_pos - window)
            end = min(len(indices), center_pos + window + 1)
            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue
                pairs.append((center, indices[context_pos]))

    if not pairs:
        return {node: target_embed.weight[i].tolist() for node, i in index_by_node.items()}

    optimizer = torch.optim.Adam(
        list(target_embed.parameters()) + list(context_embed.parameters()), lr=learning_rate
    )
    pairs_tensor = torch.tensor(pairs, dtype=torch.long)
    batch_size = min(512, len(pairs))

    for _epoch in range(epochs):
        perm = torch.randperm(pairs_tensor.shape[0])
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            batch = pairs_tensor[batch_idx]
            centers, contexts = batch[:, 0], batch[:, 1]

            center_vecs = target_embed(centers)
            context_vecs = context_embed(contexts)
            positive_score = (center_vecs * context_vecs).sum(dim=1)
            positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-10)

            negative_idx = torch.multinomial(neg_distribution, negative_samples * len(centers), replacement=True)
            negative_idx = negative_idx.view(len(centers), negative_samples)
            negative_vecs = context_embed(negative_idx)
            negative_score = torch.bmm(negative_vecs, center_vecs.unsqueeze(2)).squeeze(2)
            negative_loss = -torch.log(torch.sigmoid(-negative_score) + 1e-10).sum(dim=1)

            loss = (positive_loss + negative_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {node: target_embed.weight[i].detach().tolist() for node, i in index_by_node.items()}


def save_embeddings(embeddings: dict[str, list[float]], path: Path | str = DEFAULT_EMBEDDINGS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(embeddings), encoding="utf-8")


def load_embeddings(path: Path | str = DEFAULT_EMBEDDINGS_PATH) -> dict[str, list[float]]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def nearest_cases(
    seed_case_id: str, embeddings: dict[str, list[float]], top_k: int = 5
) -> list[tuple[str, float]]:
    """Structural nearest neighbors of a case already in the graph, by
    cosine similarity of their node2vec vectors."""
    if seed_case_id not in embeddings:
        return []
    seed_vector = embeddings[seed_case_id]
    scored = [
        (case_id, _cosine(seed_vector, vector))
        for case_id, vector in embeddings.items()
        if case_id != seed_case_id
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
