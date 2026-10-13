from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .providers.checkpoints import BASE_MODELS, MELD_ID, LAYA_ID


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    provider: Literal["editlens", "meld", "laya", "http"] = "http"
    task: Literal["text", "image"] = "text"
    model_id: str = Field(min_length=1, max_length=200)
    base_model_id: str | None = None
    endpoint: str | None = Field(default=None, max_length=2048)
    api_key: str | None = Field(default=None, max_length=4096, exclude=True)
    clear_api_key: bool = False
    enabled: bool = False
    lower_threshold: float = Field(default=.2, ge=0, lt=1)
    upper_threshold: float = Field(default=.8, gt=0, le=1)

    @model_validator(mode="after")
    def valid(self):
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("The lower threshold must be less than the upper threshold")
        if self.provider == "http" and not self.endpoint:
            raise ValueError("HTTP models require an endpoint")
        if self.provider == "laya":
            if self.task != "text" or self.model_id != LAYA_ID:
                raise ValueError("Laya supports only the reviewed local English checkpoint")
            self.base_model_id = None
        if self.provider == "meld":
            if self.task != "text" or self.model_id != MELD_ID:
                raise ValueError("MELD supports only the reviewed local MELD v5 text checkpoint")
            self.base_model_id = None
        if self.provider == "editlens" and self.task != "text":
            raise ValueError("EditLens supports text only")
        if self.provider == "editlens" and self.model_id not in BASE_MODELS:
            raise ValueError("Use the HTTP provider for other models")
        if self.provider == "editlens":
            expected = BASE_MODELS[self.model_id]
            if self.base_model_id and self.base_model_id != expected:
                raise ValueError("Base model does not match this EditLens checkpoint")
            self.base_model_id = expected
        return self


class ScanRequest(StrictModel):
    text: str = Field(min_length=1, max_length=500_000)
    title: str | None = Field(default=None, max_length=200)
    model_id: str | None = None
    check_plagiarism: bool = False


class URLScanRequest(StrictModel):
    url: str = Field(max_length=2048)
    title: str | None = Field(default=None, max_length=200)
    model_id: str | None = None
    check_plagiarism: bool = False


class BatchRequest(StrictModel):
    documents: list[ScanRequest] = Field(min_length=1, max_length=100)


class ScanPatch(StrictModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=50_000)
    feedback: Literal["helpful", "unhelpful"] | None = None


class KeyRequest(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[Literal["read", "scan"]] = Field(default=["read", "scan"], min_length=1)


class DefaultRequest(StrictModel):
    model_id: str


class ShareRequest(StrictModel):
    expires_in_hours: int = Field(default=24, ge=1, le=720)


class CorpusRequest(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=20, max_length=500_000)
    source_url: str | None = Field(default=None, max_length=2048)


class Prediction(StrictModel):
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    segments: list[dict] = Field(default_factory=list, max_length=5000)
    metadata: dict = Field(default_factory=dict)


class PaperBulkRequest(StrictModel):
    model_id: str
    q: str = Field(default='', max_length=1000)
    conference: str = Field(default='', max_length=100)
    year: int | None = None
    classified_by: str = Field(default='', max_length=200)


class PaperBulkStop(StrictModel):
    mode: Literal['finish', 'discard']
