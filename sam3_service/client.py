import itertools
import threading
from typing import Dict, List, Literal, Optional

import requests


class Sam3ServiceClient:
    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        return resp.status_code == 200

    def predict(
        self,
        image_path: str,
        prompts: List[str],
        return_masks: bool = False,
        mask_format: Literal["rle", "png"] = "rle",
        score_threshold: Optional[float] = None,
        epsilon_factor: Optional[float] = None,
        min_area: Optional[int] = None,
    ) -> Dict:
        payload = {
            "image_path": image_path,
            "prompts": prompts,
            "return_masks": return_masks,
            "mask_format": mask_format,
        }
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if epsilon_factor is not None:
            payload["epsilon_factor"] = epsilon_factor
        if min_area is not None:
            payload["min_area"] = min_area

        resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


class Sam3ServicePool:
    def __init__(self, endpoints: List[str], timeout: int = 120) -> None:
        if len(endpoints) == 0:
            raise ValueError("At least one endpoint is required")
        self.clients = [Sam3ServiceClient(url, timeout=timeout) for url in endpoints]
        self._lock = threading.Lock()
        self._cursor = itertools.cycle(range(len(self.clients)))

    def predict(self, *args, **kwargs) -> Dict:
        with self._lock:
            client_index = next(self._cursor)
        return self.clients[client_index].predict(*args, **kwargs)

    def health(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        for client in self.clients:
            try:
                status[client.base_url] = client.health()
            except Exception:
                status[client.base_url] = False
        return status
