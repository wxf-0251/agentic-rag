from typing import List


def extract_completed_sections(node_output: dict) -> List[dict]:
    """Return every completed section from a writer node update."""
    sections = node_output.get("completed_sections") or []
    latest_by_index = {}
    unindexed = []
    for position, section in enumerate(sections):
        if "section_index" not in section:
            unindexed.append((position, section))
            continue
        latest_by_index[section["section_index"]] = (position, section)

    sorted_indexed = [
        item[1]
        for _, item in sorted(latest_by_index.items(), key=lambda pair: pair[0])
    ]
    sorted_unindexed = [section for _, section in sorted(unindexed, key=lambda pair: pair[0])]
    return sorted_indexed + sorted_unindexed
