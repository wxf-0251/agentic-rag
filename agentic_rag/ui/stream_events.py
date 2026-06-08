from typing import Any, Optional, Tuple


def extract_node_update(event: dict) -> Optional[Tuple[str, dict]]:
    """Return a node update from a LangGraph stream event, or None for routing events."""
    if not event:
        return None

    node_name = list(event.keys())[0]
    node_output = event[node_name]
    if node_output is None:
        return None

    return node_name, node_output
