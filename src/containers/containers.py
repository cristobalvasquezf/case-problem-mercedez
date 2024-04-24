from pydantic import BaseModel


class GridSearchResult(BaseModel):
    best_aic: float
    best_bic: float
    best_params: dict
    best_model: object
