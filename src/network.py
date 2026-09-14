import networkx as nx
from collections import Counter

def build_transition_graph(records: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    edge_counts = Counter()

    for r in records:
        stages = r.get("stages") or []
        stages = sorted(stages, key=lambda s: s["stage_order"])
        occs = [s["occupation_clean"] for s in stages if s.get("occupation_clean")]

        for a, b in zip(occs, occs[1:]):
            edge_counts[(a, b)] += 1

    for (a, b), w in edge_counts.items():
        G.add_edge(a, b, weight=w)

    return G
