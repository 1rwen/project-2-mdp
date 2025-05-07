for k in range(1, max_iterations + 1):
        delta = 0
        v_new = v.copy()

        for s in range(NUM_STATES):
            # Bellman update for each action
            action_values = []
            for a in range(NUM_ACTIONS):
                total = 0
                for (p, s_, r, terminal) in TRANSITION_MODEL[s][a]:
                    total += p * (r + gamma * v[s_])
                action_values.append(total)

            # Value iteration update
            best_action_value = max(action_values)
            v_new[s] = best_action_value
            delta = max(delta, abs(v_new[s] - v[s]))

        v = v_new

        # Update policy from current value function
        for s in range(NUM_STATES):
            best_action = 0
            best_value = float('-inf')
            for a in range(NUM_ACTIONS):
                total = 0
                for (p, s_, r, terminal) in TRANSITION_MODEL[s][a]:
                    total += p * (r + gamma * v[s_])
                if total > best_value:
                    best_value = total
                    best_action = a
            pi[s] = best_action

        logger.log(k, v, pi)

        if delta < theta:
            break
    