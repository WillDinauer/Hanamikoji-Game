

def precompute_possibilities():
    storage = {}
    possible_ml = []
    ml = []

    def recursive_ml(idx):
        if idx == 4:
            if len(ml) > 0:
                possible_ml.append(ml.copy())
            return
        recursive_ml(idx+1)
        ml.append(idx)
        recursive_ml(idx+1)
        ml.remove(idx)

    recursive_ml(0)

    for i in range(1, 7):
        for ml in possible_ml:
            actions = []
            for move in ml:
                if move == 0:
                    for j in range(i):
                        actions.append(j)
                if move == 1:
                    for j in range(i):
                        for k in range(j+1, i):
                            actions.append((j, k))
                if move == 2:
                    for j in range(i):
                        for k in range(j+1, i):
                            for l in range(k+1, i):
                                actions.append((j, k, l))
                if move == 3:
                    for j in range(i):
                        for k in range(j+1, i):
                            for l in range(k+1, i):
                                for m in range(l+1, i):
                                    actions.append((j, k, l, m))
                                    actions.append((j, l, k, m))
                                    actions.append((j, m, k, l))
            storage[i, tuple(ml)] = actions
    return storage