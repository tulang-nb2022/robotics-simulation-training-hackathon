import pygame
import numpy as np
from dataclasses import dataclass
from typing import Tuple


Color = Tuple[int, int, int]


@dataclass
class ExplorationParams:
    action_noise: float
    step_size: int
    curiosity_weight: float


class GridWorld:
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        cell_size: int = 16,
        params: ExplorationParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.params = params or ExplorationParams(
            action_noise=0.1, step_size=1, curiosity_weight=1.0
        )

        self.rng = np.random.default_rng(seed)

        pygame.init()
        self.surface = pygame.Surface(
            (self.width * self.cell_size, self.height * self.cell_size)
        )

        self.reset()

    def reset(self) -> None:
        self.agent_pos = np.array(
            [self.width // 2, self.height // 2], dtype=np.int32
        )
        self.visits = np.zeros((self.height, self.width), dtype=np.int32)
        self._visit_current()

    def _visit_current(self) -> None:
        x, y = self.agent_pos
        self.visits[y, x] += 1

    def step(self, base_direction: np.ndarray) -> None:
        """Move agent one step with noise and curiosity bias."""
        # 4-connected moves
        moves = np.array(
            [[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32
        )

        # Choose intended move index from base_direction
        # base_direction is a vector in R^2; pick nearest move
        dots = moves @ base_direction
        idx = int(np.argmax(dots))

        # With some probability, pick a random move (exploration noise)
        if self.rng.random() < self.params.action_noise:
            idx = int(self.rng.integers(0, len(moves)))

        move = moves[idx] * self.params.step_size

        # Curiosity: bias towards less-visited neighbors
        if self.params.curiosity_weight > 0:
            visit_counts = []
            for m in moves:
                nx = int(
                    np.clip(self.agent_pos[0] + m[0], 0, self.width - 1)
                )
                ny = int(
                    np.clip(self.agent_pos[1] + m[1], 0, self.height - 1)
                )
                visit_counts.append(self.visits[ny, nx])
            visit_counts = np.array(visit_counts, dtype=np.float32)
            # Lower visits -> higher score
            scores = -visit_counts * self.params.curiosity_weight
            # Mix scores with dot alignment
            scores += dots
            idx = int(np.argmax(scores))
            move = moves[idx] * self.params.step_size

        new_pos = self.agent_pos + move
        new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)
        self.agent_pos = new_pos
        self._visit_current()

    def render(self) -> pygame.Surface:
        """Render visits as a heatmap-like grid with agent on top."""
        max_visits = np.max(self.visits)
        max_visits = max(max_visits, 1)

        self.surface.fill((10, 10, 10))

        for y in range(self.height):
            for x in range(self.width):
                v = self.visits[y, x] / max_visits
                color: Color = (
                    int(30 + 220 * v),
                    int(50 + 80 * v),
                    int(80 + 60 * (1 - v)),
                )
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.surface, color, rect)

        # Draw agent
        ax = int(self.agent_pos[0] * self.cell_size + self.cell_size / 2)
        ay = int(self.agent_pos[1] * self.cell_size + self.cell_size / 2)
        pygame.draw.circle(self.surface, (255, 255, 255), (ax, ay), self.cell_size // 3)

        return self.surface

