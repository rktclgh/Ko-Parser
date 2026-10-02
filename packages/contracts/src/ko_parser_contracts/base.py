"""모든 계약 모델의 공통 설정."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "0.2"


class ContractModel(BaseModel):
    """불변, 모르는 필드 거부, NaN·무한대 거부."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class VersionedModel(ContractModel):
    """단독으로 주고받는 루트 객체. 0.x 동안 schema_version이 정확히 같아야 읽는다."""

    schema_version: Literal["0.2"] = SCHEMA_VERSION
