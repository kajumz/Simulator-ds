from typing import List


def extend_matches(groups: List[tuple]) -> List[tuple]:
    """Extend and unite groups based on existing ones

    Example:
    input: [(1, 2), (2, 3), (5, 3), (4, 6), (6, 7), (8, 9)]
    output: [(1, 2, 3, 5), (4, 6, 7), (8, 9)]

    """
    groups = [set(group) for group in groups]
    new_groups = []

    for group in groups:
        for new_group in new_groups:
            if group & new_group:
                new_group |= group
                break
        else:
            new_groups.append(group)

    new_groups = [tuple(group) for group in new_groups]
    return new_groups