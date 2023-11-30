def get_possible_actions(state, storage, player):
        # print(f"getting possible actions for {len(player.hand), tuple(player.moves_left)}")
        # Determine the available actions based on the current state

        # Response action
        if len(state['board']['response_buffer']) > 0:
            # Possible actions are the indices of choice
            if len(state['board']['response_buffer']) == 4:
                return [[0], [1]]
            else:
                return [[0], [1], [2]]

        # Standard action
        possible_actions = storage[len(player.hand), tuple(player.moves_left)]

        # print(f"got possible actions: {possible_actions}")

        return possible_actions

def precompute_possibilities():
    storage = {}
    possible_ml = []
    ml = []

    def recursive_ml(idx):
        if idx == 5:
            if len(ml) > 0:
                possible_ml.append(ml.copy())
            return
        recursive_ml(idx+1)
        ml.append(idx)
        recursive_ml(idx+1)
        ml.remove(idx)

    recursive_ml(1)

    for i in range(1, 8):
        for ml in possible_ml:
            actions = []
            for move in ml:
                if move == 1:
                    for j in range(i):
                        actions.append([j])
                if move == 2:
                    for j in range(i):
                        for k in range(j+1, i):
                            actions.append((j, k))
                if move == 3:
                    for j in range(i):
                        for k in range(j+1, i):
                            for l in range(k+1, i):
                                actions.append((j, k, l))
                if move == 4:
                    for j in range(i):
                        for k in range(j+1, i):
                            for l in range(k+1, i):
                                for m in range(l+1, i):
                                    actions.append((j, k, l, m))
                                    actions.append((j, l, k, m))
                                    actions.append((j, m, k, l))
            storage[i, tuple(ml)] = actions
    return storage


def create_action_dict():
    action_dict = {}
    int_dict = {}

    action = 0
    for i in range(7):
        action_dict[i] = action
        int_dict[action] = [i]
        action += 1

    for i in range(7):
        for j in range(i+1, 7):
            action_dict[(i, j)] = action
            int_dict[action] = (i, j)
            action += 1

    for i in range(7):
        for j in range(i+1, 7):
            for k in range(j+1, 7):
                action_dict[(i, j, k)] = action
                int_dict[action] = (i, j, k)
                action += 1

    for i in range(7):
        for j in range(i, 7):
            for k in range(j+1, 7):
                for m in range(k+1, 7):
                    action_dict[(i, j, k, m)] = action
                    action_dict[(i, m, j, k)] = action+1
                    action_dict[(i, k, j, m)] = action+2

                    int_dict[action] = (i, j, k, m)
                    int_dict[action+1] = (i, m, j, k)
                    int_dict[action+2] = (i, k, j, m)
                    action += 3

    return action_dict, int_dict