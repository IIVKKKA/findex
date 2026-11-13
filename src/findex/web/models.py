"""Pydantic models live only at the HTTP boundary; inside, we use the dataclasses of labs 2-3."""

from typing import Literal

from pydantic import BaseModel, Field

ScorerName = Literal["bm25", "tfidf"]
Mode = Literal["lexical", "semantic", "hybrid"]


class HitOut(BaseModel):
    doc_id: int
    score: float
    title: str
    snippet: str = Field(description="Text window around the best match; matches wrapped in **...**")


class SearchOut(BaseModel):
    query: str
    scorer: ScorerName
    mode: Mode = "lexical"
    page: int
    per_page: int
    total: int = Field(description="Number of documents matching the query")
    pages: int
    results: list[HitOut]


class DocOut(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int = Field(description="Length in tokens")
    text: str


class StatsOut(BaseModel):
    documents: int
    terms: int
    avg_doc_length: float
    layout: str
    positions: bool


class HealthOut(BaseModel):
    status: Literal["ok", "loading"]
    documents: int = 0


class ErrorOut(BaseModel):
    detail: str
    request_id: str | None = None
