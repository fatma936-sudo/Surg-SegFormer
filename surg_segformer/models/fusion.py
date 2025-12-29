from __future__ import annotations
import torch
from .common import logits_to_pred, probs_max

class PriorityWeightedFusion:
    """
    Priority-weighted conditional fusion.

    This repo trains *two* models on *disjoint label spaces*:
      - tools model: background + tool classes (contiguous ids)
      - anatomy model: background + anatomy classes (contiguous ids)

    Therefore, at inference we map both predictions back to the global label ids (labels.json),
    then overwrite anatomy with tools where the tools prediction is confident.

    Config keys:
      - tool_background_id_global: e.g., 0
      - conf_threshold: e.g., 0.5
      - tools_local_to_global: dict[int,int] (optional if your tools model outputs global ids already)
      - anatomy_local_to_global: dict[int,int] (optional if anatomy model outputs global ids already)
    """
    def __init__(
        self,
        tool_background_id_global: int = 0,
        conf_threshold: float = 0.5,
        tools_local_to_global: dict[int, int] | None = None,
        anatomy_local_to_global: dict[int, int] | None = None,
    ):
        self.tool_background_id_global = int(tool_background_id_global)
        self.conf_threshold = float(conf_threshold)
        self.tools_local_to_global = tools_local_to_global or {}
        self.anatomy_local_to_global = anatomy_local_to_global or {}

    @staticmethod
    def _map_ids(x: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
        if not mapping:
            return x
        out = torch.empty_like(x)
        # default: keep as-is
        out.copy_(x)
        for src, dst in mapping.items():
            out = torch.where(x == int(src), torch.tensor(int(dst), device=x.device, dtype=x.dtype), out)
        return out

    @torch.no_grad()
    def __call__(self, anatomy_logits: torch.Tensor, tools_logits: torch.Tensor) -> torch.Tensor:
        a_pred = logits_to_pred(anatomy_logits)  # local or global ids
        t_pred = logits_to_pred(tools_logits)
        t_conf = probs_max(tools_logits)

        # map to global ids if needed
        a_global = self._map_ids(a_pred, self.anatomy_local_to_global)
        t_global = self._map_ids(t_pred, self.tools_local_to_global)

        tool_mask = (t_global != self.tool_background_id_global) & (t_conf >= self.conf_threshold)
        final = torch.where(tool_mask, t_global, a_global)
        return final
