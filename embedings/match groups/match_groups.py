from typing import List
from typing import Tuple
#n = int(input())
#l = []
#for i in range(n+1):
#    my_tuple = tuple(map(int, input().split()))
#    print(my_tuple)
#    l.append(my_tuple)
#rea = l[1:]

def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """add new pairs"""
    # unique id
    un = set()
    for pair in pairs:
        un.add(pair[0])
        un.add(pair[1])
    # create dict with exist pair
    pairs_dict = {x : {x} for x in un}
    for x, y in pairs:
        united_pairs = pairs_dict[x] | pairs_dict[y]
        for z in united_pairs:
            pairs_dict[z].update(united_pairs)
    new_pairs = []
    print(pairs_dict.items())
    for x, x_pairs in pairs_dict.items():
        for y in x_pairs:
            new_pairs.append((x,y))
    new_pairs = [pair for pair in new_pairs if pair[0] < pair[1]]
    new_pairs = sorted(new_pairs)
    return new_pairs

#print(extend_matches(rea))
#extend_matches(rea)