from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .search_space import SearchSpace


@dataclass
class SearchResult:
    best_config: Dict[str, object]
    best_score: float
    history: List[Dict[str, object]]
    total_time_sec: float


def _weights_from_scores(scores: Sequence[float]) -> List[float]:
    vals = [1.0 / max(1e-12, float(s)) for s in scores]
    total = sum(vals)
    if total <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [v / total for v in vals]


class BaseSearchOptimizer:
    name = "base"

    def __init__(
        self,
        budget: int,
        seed: int,
        colony_size: int = 4,
        limit: int = 3,
        elite_ratio: float = 0.4,
        stage_name: str = "stage",
        display_name: Optional[str] = None,
    ) -> None:
        self.budget = int(budget)
        self.seed = int(seed)
        self.colony_size = int(colony_size)
        self.limit = int(limit)
        self.elite_ratio = float(elite_ratio)
        self.stage_name = stage_name
        self.display_name = str(display_name or self.name).upper()
        self.rng = random.Random(self.seed)

    def _evaluate(
        self,
        objective,
        space: SearchSpace,
        vector: Sequence[float],
        model_name: str,
        phase: str,
    ):
        config = space.decode_vector(vector, model_name=model_name)
        result = objective.evaluate(config, phase=phase)
        return result.score, result.config, result.metrics

    def _log_eval_start(self, eval_index: int, label: str) -> None:
        print(
            f"[Search][{self.display_name}][{self.stage_name}] {label} | eval {eval_index}/{self.budget}",
            flush=True,
        )

    def _log_new_best(self, eval_index: int, score: float) -> None:
        print(
            f"[Search][{self.display_name}][{self.stage_name}] new best | eval {eval_index}/{self.budget} | val_mse={float(score):.6f}",
            flush=True,
        )

    def _log_search_complete(self, eval_count: int, best_score: float) -> None:
        print(
            f"[Search][{self.display_name}][{self.stage_name}] complete | evals={eval_count}/{self.budget} | best_val_mse={float(best_score):.6f}",
            flush=True,
        )

    def search(
        self,
        objective,
        space: SearchSpace,
        model_name: str,
        center: Optional[Sequence[float]] = None,
        active_names: Optional[Iterable[str]] = None,
        radius: Optional[float] = None,
    ) -> SearchResult:
        raise NotImplementedError


class RandomSearchOptimizer(BaseSearchOptimizer):
    name = "random_search"

    def search(self, objective, space, model_name, center=None, active_names=None, radius=None):
        start = time.time()
        best_score = float("inf")
        best_config = None
        history: List[Dict[str, object]] = []

        center = list(center or space.encode_config(space.get_default_config(model_name)))
        initial_vectors = [center]
        for _ in range(max(0, self.budget - 1)):
            initial_vectors.append(
                space.sample_vector(
                    self.rng,
                    model_name=model_name,
                    center=center,
                    active_names=active_names,
                    radius=radius,
                )
            )
        for step, vec in enumerate(initial_vectors, start=1):
            self._log_eval_start(step, f"step {step}")
            score, config, metrics = self._evaluate(
                objective, space, vec, model_name, phase=f"{self.stage_name}_random"
            )
            previous_best = best_score
            if score < best_score:
                best_score = score
                best_config = config
            if best_score < previous_best:
                self._log_new_best(step, best_score)
            history.append(
                {
                    "step": step,
                    "score": score,
                    "best_score": best_score,
                    "config": config,
                    "phase": self.stage_name,
                    **metrics,
                }
            )

        self._log_search_complete(len(history), best_score)
        return SearchResult(
            best_config=best_config,
            best_score=best_score,
            history=history,
            total_time_sec=time.time() - start,
        )


class GeneticSearchOptimizer(BaseSearchOptimizer):
    name = "ga"

    def _crossover(self, a, b):
        child = list(a)
        for i in range(len(child)):
            if self.rng.random() < 0.5:
                child[i] = b[i]
        return child

    def _mutate(self, child, space, center, active_names, radius):
        active_names = list(active_names or space.names)
        idx = self.rng.choice([space.names.index(n) for n in active_names])
        child[idx] = min(1.0, max(0.0, child[idx] + self.rng.uniform(-0.2, 0.2)))
        return space.clip_vector(child, center=center, active_names=active_names, radius=radius)

    def search(self, objective, space, model_name, center=None, active_names=None, radius=None):
        start = time.time()
        active_names = list(active_names or space.get_active_names(model_name))
        center = list(center or space.encode_config(space.get_default_config(model_name)))
        pop = [list(center)]
        pop.extend(
            space.sample_vector(self.rng, model_name, center=center, active_names=active_names, radius=radius)
            for _ in range(max(0, self.colony_size - 1))
        )
        scores, configs = [], []
        history = []
        eval_count = 0
        best_score = float("inf")
        best_config = None

        for i in range(len(pop)):
            self._log_eval_start(eval_count + 1, f"init member {i + 1}/{len(pop)}")
            score, config, metrics = self._evaluate(objective, space, pop[i], model_name, f"{self.stage_name}_ga_init")
            eval_count += 1
            scores.append(score)
            configs.append(config)
            previous_best = best_score
            if score < best_score:
                best_score = score
                best_config = config
            if best_score < previous_best:
                self._log_new_best(eval_count, best_score)
            history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})
            if eval_count >= self.budget:
                break

        generation = 1
        while eval_count < self.budget:
            ranked_idx = sorted(range(len(pop)), key=lambda i: scores[i])
            parents = [pop[i] for i in ranked_idx[: max(2, len(pop) // 2)]]
            children = []
            while len(children) < len(pop) and eval_count + len(children) < self.budget:
                pa, pb = self.rng.sample(parents, 2)
                child = self._mutate(self._crossover(pa, pb), space, center, active_names, radius)
                children.append(child)
            for child_idx, child in enumerate(children, start=1):
                self._log_eval_start(eval_count + 1, f"generation {generation} | child {child_idx}/{len(children)}")
                score, config, metrics = self._evaluate(objective, space, child, model_name, f"{self.stage_name}_ga")
                eval_count += 1
                worst = max(range(len(pop)), key=lambda i: scores[i])
                if score < scores[worst]:
                    pop[worst], scores[worst], configs[worst] = child, score, config
                previous_best = best_score
                if score < best_score:
                    best_score = score
                    best_config = config
                if best_score < previous_best:
                    self._log_new_best(eval_count, best_score)
                history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})
                if eval_count >= self.budget:
                    break
            generation += 1

        self._log_search_complete(eval_count, best_score)
        return SearchResult(best_config=best_config, best_score=best_score, history=history, total_time_sec=time.time() - start)


class PSOSearchOptimizer(BaseSearchOptimizer):
    name = "pso"

    def search(self, objective, space, model_name, center=None, active_names=None, radius=None):
        start = time.time()
        active_names = list(active_names or space.get_active_names(model_name))
        center = list(center or space.encode_config(space.get_default_config(model_name)))

        particles = [list(center)]
        particles.extend(
            space.sample_vector(self.rng, model_name, center=center, active_names=active_names, radius=radius)
            for _ in range(max(0, self.colony_size - 1))
        )
        velocities = [[0.0] * len(space.names) for _ in range(self.colony_size)]
        pbest = [list(p) for p in particles]
        pbest_scores = [float("inf")] * self.colony_size
        gbest = None
        gbest_score = float("inf")
        gbest_config = None
        history = []
        eval_count = 0
        iteration = 1

        while eval_count < self.budget:
            for i in range(self.colony_size):
                if eval_count >= self.budget:
                    break
                self._log_eval_start(eval_count + 1, f"iteration {iteration} | particle {i + 1}/{self.colony_size}")
                score, config, metrics = self._evaluate(objective, space, particles[i], model_name, f"{self.stage_name}_pso")
                eval_count += 1
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = list(particles[i])
                previous_best = gbest_score
                if score < gbest_score:
                    gbest_score = score
                    gbest = list(particles[i])
                    gbest_config = config
                if gbest_score < previous_best:
                    self._log_new_best(eval_count, gbest_score)
                history.append({"step": eval_count, "score": score, "best_score": gbest_score, "config": config, "phase": self.stage_name, **metrics})

            if gbest is None:
                break

            for i in range(self.colony_size):
                for dim, name in enumerate(space.names):
                    if name not in active_names:
                        particles[i][dim] = center[dim]
                        velocities[i][dim] = 0.0
                        continue
                    r1 = self.rng.random()
                    r2 = self.rng.random()
                    velocities[i][dim] = (
                        0.5 * velocities[i][dim]
                        + 1.5 * r1 * (pbest[i][dim] - particles[i][dim])
                        + 1.5 * r2 * (gbest[dim] - particles[i][dim])
                    )
                    particles[i][dim] += velocities[i][dim]
                particles[i] = space.clip_vector(particles[i], center=center, active_names=active_names, radius=radius)
            iteration += 1

        self._log_search_complete(eval_count, gbest_score)
        return SearchResult(best_config=gbest_config, best_score=gbest_score, history=history, total_time_sec=time.time() - start)


class OriginalABCSearchOptimizer(BaseSearchOptimizer):
    name = "abc"

    def _mutate(self, x_i, x_k, space, center, active_names, radius):
        active_names = list(active_names or space.names)
        v = list(x_i)
        dim = self.rng.choice([space.names.index(n) for n in active_names])
        phi = self.rng.uniform(-1.0, 1.0)
        v[dim] = x_i[dim] + phi * (x_i[dim] - x_k[dim])
        return space.clip_vector(v, center=center, active_names=active_names, radius=radius)

    def search(self, objective, space, model_name, center=None, active_names=None, radius=None):
        start = time.time()
        active_names = list(active_names or space.get_active_names(model_name))
        center = list(center or space.encode_config(space.get_default_config(model_name)))
        food_sources = [list(center)]
        food_sources.extend(
            space.sample_vector(self.rng, model_name, center=center, active_names=active_names, radius=radius)
            for _ in range(max(0, self.colony_size - 1))
        )
        scores, configs, trials = [], [], [0] * self.colony_size
        history = []
        best_score = float("inf")
        best_config = None
        eval_count = 0

        # init
        for i in range(self.colony_size):
            self._log_eval_start(eval_count + 1, f"init source {i + 1}/{self.colony_size}")
            score, config, metrics = self._evaluate(objective, space, food_sources[i], model_name, f"{self.stage_name}_abc_init")
            eval_count += 1
            scores.append(score)
            configs.append(config)
            previous_best = best_score
            if score < best_score:
                best_score = score
                best_config = config
            if best_score < previous_best:
                self._log_new_best(eval_count, best_score)
            history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})
            if eval_count >= self.budget:
                break

        cycle = 1
        while eval_count < self.budget:
            # employed bees
            for i in range(self.colony_size):
                if eval_count >= self.budget:
                    break
                self._log_eval_start(eval_count + 1, f"cycle {cycle} | employed bee {i + 1}/{self.colony_size}")
                k = self.rng.choice([j for j in range(self.colony_size) if j != i])
                cand = self._mutate(food_sources[i], food_sources[k], space, center, active_names, radius)
                score, config, metrics = self._evaluate(objective, space, cand, model_name, f"{self.stage_name}_abc_employed")
                eval_count += 1
                if score < scores[i]:
                    food_sources[i], scores[i], configs[i], trials[i] = cand, score, config, 0
                else:
                    trials[i] += 1
                previous_best = best_score
                if score < best_score:
                    best_score, best_config = score, config
                if best_score < previous_best:
                    self._log_new_best(eval_count, best_score)
                history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})

            # onlookers
            probs = _weights_from_scores(scores)
            for onlooker_idx in range(self.colony_size):
                if eval_count >= self.budget:
                    break
                self._log_eval_start(eval_count + 1, f"cycle {cycle} | onlooker bee {onlooker_idx + 1}/{self.colony_size}")
                i = self.rng.choices(range(self.colony_size), weights=probs, k=1)[0]
                k = self.rng.choice([j for j in range(self.colony_size) if j != i])
                cand = self._mutate(food_sources[i], food_sources[k], space, center, active_names, radius)
                score, config, metrics = self._evaluate(objective, space, cand, model_name, f"{self.stage_name}_abc_onlooker")
                eval_count += 1
                if score < scores[i]:
                    food_sources[i], scores[i], configs[i], trials[i] = cand, score, config, 0
                else:
                    trials[i] += 1
                previous_best = best_score
                if score < best_score:
                    best_score, best_config = score, config
                if best_score < previous_best:
                    self._log_new_best(eval_count, best_score)
                history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})

            # scouts
            for i in range(self.colony_size):
                if eval_count >= self.budget:
                    break
                if trials[i] >= self.limit:
                    self._log_eval_start(eval_count + 1, f"cycle {cycle} | scout bee {i + 1}/{self.colony_size}")
                    food_sources[i] = space.sample_vector(self.rng, model_name, center=center, active_names=active_names, radius=radius)
                    score, config, metrics = self._evaluate(objective, space, food_sources[i], model_name, f"{self.stage_name}_abc_scout")
                    eval_count += 1
                    scores[i], configs[i], trials[i] = score, config, 0
                    previous_best = best_score
                    if score < best_score:
                        best_score, best_config = score, config
                    if best_score < previous_best:
                        self._log_new_best(eval_count, best_score)
                    history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})
            cycle += 1

        self._log_search_complete(eval_count, best_score)
        return SearchResult(best_config=best_config, best_score=best_score, history=history, total_time_sec=time.time() - start)


class EGTAABCSearchOptimizer(OriginalABCSearchOptimizer):
    name = "egta_abc"

    def __init__(
        self,
        budget: int,
        seed: int,
        colony_size: int = 4,
        limit: int = 3,
        elite_ratio: float = 0.4,
        stage_name: str = "stage",
        display_name: Optional[str] = None,
        use_elite: bool = True,
        type_aware: bool = True,
        use_projection: bool = True,
    ) -> None:
        super().__init__(
            budget=budget,
            seed=seed,
            colony_size=colony_size,
            limit=limit,
            elite_ratio=elite_ratio,
            stage_name=stage_name,
            display_name=display_name,
        )
        self.use_elite = use_elite
        self.type_aware = type_aware
        self.use_projection = use_projection

    def _elite_center(self, food_sources, scores):
        pairs = sorted(zip(scores, food_sources), key=lambda x: x[0])
        k = max(1, math.ceil(len(food_sources) * self.elite_ratio))
        elites = [vec for _, vec in pairs[:k]]
        center = []
        for dim in range(len(elites[0])):
            center.append(sum(v[dim] for v in elites) / k)
        return center

    def _mutate(self, x_i, x_k, elite_center, space, center, active_names, radius):
        active_names = list(active_names or space.names)
        v = list(x_i)
        dim = self.rng.choice([space.names.index(n) for n in active_names])
        phi = self.rng.uniform(-1.0, 1.0)
        lam = self.rng.uniform(0.2, 0.8)
        if self.type_aware and space.is_discrete(space.names[dim]):
            phi *= 0.5
        elite_term = 0.0
        if self.use_elite and elite_center is not None:
            elite_term = lam * (elite_center[dim] - x_i[dim])
        v[dim] = x_i[dim] + phi * (x_i[dim] - x_k[dim]) + elite_term
        if self.use_projection:
            return space.clip_vector(v, center=center, active_names=active_names, radius=radius)
        return [min(1.0, max(0.0, x)) for x in v]

    def search(self, objective, space, model_name, center=None, active_names=None, radius=None):
        start = time.time()
        active_names = list(active_names or space.get_active_names(model_name))
        center = list(center or space.encode_config(space.get_default_config(model_name)))
        food_sources = [list(center)]
        food_sources.extend(
            space.sample_vector(self.rng, model_name, center=center, active_names=active_names, radius=radius)
            for _ in range(max(0, self.colony_size - 1))
        )
        scores, configs, trials = [], [], [0] * self.colony_size
        history = []
        best_score = float("inf")
        best_config = None
        eval_count = 0

        for i in range(self.colony_size):
            self._log_eval_start(eval_count + 1, f"init source {i + 1}/{self.colony_size}")
            score, config, metrics = self._evaluate(objective, space, food_sources[i], model_name, f"{self.stage_name}_egta_init")
            eval_count += 1
            scores.append(score)
            configs.append(config)
            previous_best = best_score
            if score < best_score:
                best_score, best_config = score, config
            if best_score < previous_best:
                self._log_new_best(eval_count, best_score)
            history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})
            if eval_count >= self.budget:
                break

        cycle = 1
        while eval_count < self.budget:
            elite_center = self._elite_center(food_sources, scores)

            for i in range(self.colony_size):
                if eval_count >= self.budget:
                    break
                self._log_eval_start(eval_count + 1, f"cycle {cycle} | employed bee {i + 1}/{self.colony_size}")
                k = self.rng.choice([j for j in range(self.colony_size) if j != i])
                cand = self._mutate(food_sources[i], food_sources[k], elite_center, space, center, active_names, radius)
                score, config, metrics = self._evaluate(objective, space, cand, model_name, f"{self.stage_name}_egta_employed")
                eval_count += 1
                if score < scores[i]:
                    food_sources[i], scores[i], configs[i], trials[i] = cand, score, config, 0
                else:
                    trials[i] += 1
                previous_best = best_score
                if score < best_score:
                    best_score, best_config = score, config
                if best_score < previous_best:
                    self._log_new_best(eval_count, best_score)
                history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})

            probs = _weights_from_scores(scores)
            for onlooker_idx in range(self.colony_size):
                if eval_count >= self.budget:
                    break
                self._log_eval_start(eval_count + 1, f"cycle {cycle} | onlooker bee {onlooker_idx + 1}/{self.colony_size}")
                i = self.rng.choices(range(self.colony_size), weights=probs, k=1)[0]
                k = self.rng.choice([j for j in range(self.colony_size) if j != i])
                cand = self._mutate(food_sources[i], food_sources[k], elite_center, space, center, active_names, radius)
                score, config, metrics = self._evaluate(objective, space, cand, model_name, f"{self.stage_name}_egta_onlooker")
                eval_count += 1
                if score < scores[i]:
                    food_sources[i], scores[i], configs[i], trials[i] = cand, score, config, 0
                else:
                    trials[i] += 1
                previous_best = best_score
                if score < best_score:
                    best_score, best_config = score, config
                if best_score < previous_best:
                    self._log_new_best(eval_count, best_score)
                history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})

            for i in range(self.colony_size):
                if eval_count >= self.budget:
                    break
                if trials[i] >= self.limit:
                    self._log_eval_start(eval_count + 1, f"cycle {cycle} | scout bee {i + 1}/{self.colony_size}")
                    food_sources[i] = space.sample_vector(self.rng, model_name, center=center, active_names=active_names, radius=radius)
                    score, config, metrics = self._evaluate(objective, space, food_sources[i], model_name, f"{self.stage_name}_egta_scout")
                    eval_count += 1
                    scores[i], configs[i], trials[i] = score, config, 0
                    previous_best = best_score
                    if score < best_score:
                        best_score, best_config = score, config
                    if best_score < previous_best:
                        self._log_new_best(eval_count, best_score)
                    history.append({"step": eval_count, "score": score, "best_score": best_score, "config": config, "phase": self.stage_name, **metrics})
            cycle += 1

        self._log_search_complete(eval_count, best_score)
        return SearchResult(best_config=best_config, best_score=best_score, history=history, total_time_sec=time.time() - start)
