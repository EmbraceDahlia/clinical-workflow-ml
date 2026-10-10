"""Genetic algorithm that searches blend weights and decision thresholds to maximise F1."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from workflow_ml.config import SEED


@dataclass
class GeneticAlgorithm:
    """Real-valued GA on genes in [0, 1]: tournament selection, uniform crossover, Gaussian mutation."""

    population_size: int = 60
    generations: int = 50
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    mutation_scale: float = 0.08
    elite: int = 2
    seed: int = SEED

    def maximise(
        self, fitness: Callable[[np.ndarray], float], n_genes: int,
        seeds: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[float]]:
        rng = np.random.default_rng(self.seed)
        population = rng.random((self.population_size, n_genes))
        if seeds is not None:
            population[: len(seeds)] = seeds
        history: list[float] = []

        for _ in range(self.generations):
            scores = np.array([fitness(chromosome) for chromosome in population])
            ranked = population[np.argsort(-scores)]
            history.append(float(scores.max()))

            children = [ranked[i].copy() for i in range(self.elite)]
            while len(children) < self.population_size:
                mother, father = (self._select(population, scores, rng) for _ in range(2))
                child = mother.copy()
                if rng.random() < self.crossover_rate:
                    from_father = rng.random(n_genes) < 0.5
                    child[from_father] = father[from_father]
                mutate = rng.random(n_genes) < self.mutation_rate
                child[mutate] += rng.normal(0, self.mutation_scale, mutate.sum())
                children.append(np.clip(child, 0, 1))
            population = np.array(children)

        scores = np.array([fitness(chromosome) for chromosome in population])
        history.append(float(scores.max()))
        return population[np.argmax(scores)], history

    def _select(self, population: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        contenders = rng.integers(0, len(population), self.tournament_size)
        return population[contenders[np.argmax(scores[contenders])]]


@dataclass
class Blend:
    weights: np.ndarray      # (n_labels, n_models), each row sums to 1
    thresholds: np.ndarray   # (n_labels,)
    history: list[list[float]]

    def scores(self, model_scores: np.ndarray) -> np.ndarray:
        """`model_scores` has shape (n_rows, n_models, n_labels)."""
        return np.einsum("rml,lm->rl", model_scores, self.weights)


def _f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    return 2 * (truth & predicted).sum() / max(truth.sum() + predicted.sum(), 1)


def evolve_blend(model_scores: np.ndarray, y_true: np.ndarray, ga: GeneticAlgorithm | None = None) -> Blend:
    """Per label, evolve a chromosome = [one weight per model..., decision threshold]."""
    ga = ga or GeneticAlgorithm()
    n_models, n_labels = model_scores.shape[1:]
    weights, thresholds, histories = [], [], []

    for label in range(n_labels):
        scores, truth = model_scores[:, :, label], y_true[:, label].astype(bool)

        def decode(chromosome: np.ndarray) -> tuple[np.ndarray, float]:
            weight = chromosome[:n_models] + 1e-9
            return weight / weight.sum(), float(chromosome[-1])

        def fitness(chromosome: np.ndarray) -> float:
            weight, threshold = decode(chromosome)
            return _f1(truth, scores @ weight >= threshold)

        # Start some chromosomes from the single-model solutions so the GA never does worse.
        seeds = np.hstack([np.eye(n_models), np.full((n_models, 1), 0.5)])
        best, history = ga.maximise(fitness, n_models + 1, seeds)
        weight, threshold = decode(best)
        weights.append(weight)
        thresholds.append(threshold)
        histories.append(history)

    return Blend(np.array(weights), np.array(thresholds), histories)
