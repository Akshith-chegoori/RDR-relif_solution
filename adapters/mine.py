"""Fast 2D bin-packing heuristic for Relief Loading."""
from adapter import Solver
import math
import random
import time


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        n = instance.size
        W = instance.max_weight
        V = instance.max_volume
        w = instance.weights
        v = instance.volumes
        deadline = time.perf_counter() + 4.70

        best_cost = 10**18
        best_plan = None

        def cost(plan):
            least = 1000
            used = 0
            for b in plan:
                if not b:
                    continue
                sw = sum(w[p] for p in b)
                sv = sum(v[p] for p in b)
                if sw > W or sv > V:
                    return 10**18
                used += 1
                least = min(least, max(sw * 1000 // W, sv * 1000 // V))
            return used * 1000 + least if used else 10**18

        def submit(plan):
            nonlocal best_cost, best_plan
            if time.perf_counter() >= deadline:
                return False
            c = cost(plan)
            if c >= best_cost:
                return False
            r = submit_candidate({"trucks": plan})
            if r.get("accepted"):
                rc = r.get("best")
                if rc is not None and rc < best_cost:
                    best_cost = rc
                    best_plan = [b[:] for b in plan]
                    return True
            return False

        # Very fast safety candidate: ordinary 2D best-fit decreasing.
        order = sorted(
            range(n),
            key=lambda p: max(w[p] / W, v[p] / V),
            reverse=True,
        )
        bins = []
        loads = []
        for p in order:
            pick = None
            pick_score = -1.0
            for j, (sw, sv) in enumerate(loads):
                nw = sw + w[p]
                nv = sv + v[p]
                if nw <= W and nv <= V:
                    s = max(nw / W, nv / V) - 0.04 * abs(nw / W - nv / V)
                    if s > pick_score:
                        pick_score = s
                        pick = j
            if pick is None:
                bins.append([p])
                loads.append((w[p], v[p]))
            else:
                bins[pick].append(p)
                loads[pick] = (loads[pick][0] + w[p], loads[pick][1] + v[p])
        submit(bins)

        # The generator creates the data from near-full witness trucks.  The
        # aggregate lower bound is therefore an excellent target for the
        # number of trucks.  Complete one very-full truck at a time, leaving
        # the remainder for the final truck.
        lower_bound = max(
            math.ceil(sum(w) / W),
            math.ceil(sum(v) / V),
        )

        def completion_pack(k, seed, phase_deadline=None, score_a=0.10, score_bias=0.0, threshold=0.985):
            rng = random.Random(seed)
            remaining = set(range(n))
            result = []

            # Fill k-1 trucks greedily, deliberately seeking trucks that are
            # close to full in BOTH dimensions.
            for _ in range(k - 1):
                if not remaining or time.perf_counter() >= deadline or (phase_deadline is not None and time.perf_counter() >= phase_deadline):
                    return None

                candidates = [p for p in range(n) if p in remaining]
                rng.shuffle(candidates)
                candidates.sort(
                    key=lambda p: max(w[p] / W, v[p] / V),
                    reverse=True,
                )
                candidates = candidates[:min(24, len(candidates))]

                best = None
                for first in candidates:
                    sw = w[first]
                    sv = v[first]
                    chosen = [first]
                    available = [p for p in remaining if p != first]

                    while available and time.perf_counter() < deadline and (phase_deadline is None or time.perf_counter() < phase_deadline):
                        pick = None
                        pick_score = -1e30
                        # Sample all candidates; n <= 300, so this remains
                        # cheap while being much stronger than plain FFD.
                        for p in available:
                            nw = sw + w[p]
                            nv = sv + v[p]
                            if nw <= W and nv <= V:
                                x = nw / W
                                y = nv / V
                                score = min(x, y) + score_a * max(x, y) - score_bias * abs(x - y) + rng.random() * 1e-8
                                if score > pick_score:
                                    pick_score = score
                                    pick = p
                        if pick is None:
                            break
                        chosen.append(pick)
                        available.remove(pick)
                        sw += w[pick]
                        sv += v[pick]
                        if min(sw / W, sv / V) >= threshold:
                            break

                    score = min(sw / W, sv / V) + score_a * max(sw / W, sv / V) - score_bias * abs(sw / W - sv / V)
                    if best is None or score > best[0]:
                        best = (score, chosen)

                if best is None:
                    return None
                chosen = best[1]
                result.append(chosen)
                remaining.difference_update(chosen)

            if not remaining:
                return result if len(result) == k else None

            sw = sum(w[p] for p in remaining)
            sv = sum(v[p] for p in remaining)
            if sw <= W and sv <= V:
                result.append(list(remaining))
                return result
            return None

        def fast_completion(k, seed):
            """One-start version used to get a strong candidate before 5%."""
            rng = random.Random(seed)
            remaining = set(range(n))
            result = []
            for _ in range(k - 1):
                if not remaining:
                    return None
                first = max(remaining, key=lambda p: max(w[p] / W, v[p] / V))
                chosen = [first]
                sw, sv = w[first], v[first]
                available = [p for p in remaining if p != first]
                while available:
                    pick = None
                    pick_score = -1e30
                    for p in available:
                        nw, nv = sw + w[p], sv + v[p]
                        if nw <= W and nv <= V:
                            x, y = nw / W, nv / V
                            score = min(x, y) + 0.10 * max(x, y)
                            if score > pick_score:
                                pick_score, pick = score, p
                    if pick is None:
                        break
                    chosen.append(pick)
                    available.remove(pick)
                    sw += w[pick]
                    sv += v[pick]
                    if min(sw / W, sv / V) >= 0.985:
                        break
                result.append(chosen)
                remaining.difference_update(chosen)
            sw = sum(w[p] for p in remaining)
            sv = sum(v[p] for p in remaining)
            if sw <= W and sv <= V:
                result.append(list(remaining))
                return result
            return None

        # Get a strong one-start candidate immediately. This is particularly
        # useful on large instances where the full lower-bound search can take
        # longer than the first scoring checkpoint.
        fast_extra = fast_completion(lower_bound + 1, 0)
        if fast_extra is not None:
            submit(fast_extra)
        fast_lb = fast_completion(lower_bound, 0)
        if fast_lb is not None:
            submit(fast_lb)

        # Try the aggregate lower bound first.  One extra truck is still
        # dramatically better than the old heuristic when the lower bound is
        # not attainable by this fast construction.
        for ki, k in enumerate((lower_bound, lower_bound + 1)):
            if time.perf_counter() >= deadline:
                break
            # Never let a difficult lower-bound construction consume the
            # early checkpoint. If it is hard, move immediately to one extra
            # truck; that is still far better than a weak late candidate.
            phase_deadline = min(deadline, time.perf_counter() + (0.18 if ki == 0 else 4.0))
            for seed in range(12):
                if time.perf_counter() >= phase_deadline:
                    break
                plan = completion_pack(k, seed, phase_deadline)
                if plan is not None:
                    submit(plan)
                    if k == lower_bound and time.perf_counter() > deadline - 0.7:
                        break
                # One complementary packing objective often improves the
                # secondary least-full-truck term without sacrificing the
                # primary truck-count objective.
                if seed == 0 and time.perf_counter() < phase_deadline - 0.03:
                    alt = completion_pack(k, 0, phase_deadline, 0.60, 0.06, 0.99)
                    if alt is not None:
                        submit(alt)

        if best_plan is not None:
            return {"trucks": best_plan}
        return {"trucks": bins}
