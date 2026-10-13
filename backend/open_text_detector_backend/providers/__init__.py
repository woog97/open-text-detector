from .errors import ProviderError
from .base import normalize_prediction
import json
from ..network import request_bytes, NetworkError


class Providers:
    def __init__(self, settings, db):
        self.settings, self.db = settings, db
        self._editlens = None
        self._meld = None
        self._laya = None

    def classify(self, model, text, image=None):
        if model["provider"] == "laya":
            if self._laya is None:
                from .laya import Laya
                self._laya = Laya(self.settings.model_device, self.settings.model_dir,
                                  self.settings.laya_runtime, self.settings.laya_batch_size,
                                  self.settings.laya_precision)
            return self._laya.predict(model, text)
        elif model["provider"] == "meld":
            if self._meld is None:
                from .meld import Meld
                self._meld = Meld(self.settings.model_device, self.settings.model_dir, self.settings.meld_precision, self.settings.meld_batch_size)
            return self._meld.predict(model, text)
        elif model["provider"] == "editlens":
            if self._editlens is None:
                from .editlens import EditLens
                self._editlens = EditLens(self.settings.model_device, self.settings.hf_token.get_secret_value() if self.settings.hf_token else None, self.settings.model_dir)
            raw = self._editlens.predict(model, text)
        elif model["provider"] == "http":
            payload = {"model": model["model_id"], "text": text} if image is None else {
                "model": model["model_id"], "image_base64": image}
            try:
                body, _, _ = request_bytes(model["endpoint"], payload=payload,
                    token=self.db.decrypt(model.get("secret")), model_endpoint=True,
                    allow_local=self.settings.allow_local_model_endpoints, timeout=self.settings.provider_timeout)
                raw = json.loads(body)
                if not isinstance(raw, dict):
                    raise ValueError()
            except NetworkError as e:
                raise ProviderError("provider_unavailable", str(e)) from e
            except (ValueError, TypeError) as e:
                raise ProviderError("invalid_response", "Provider did not return a JSON object") from e
        else:
            raise ProviderError("unknown_provider", "Unsupported provider")
        result = normalize_prediction(raw, text, model)
        if model["provider"] == "editlens":
            result["inference"] = raw["inference"]
        return result
