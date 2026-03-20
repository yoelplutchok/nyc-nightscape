"""
Schema validation for canonical outputs.

- boro_cd is always pandas nullable Int64.
- Schema validation is mandatory on read and write.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

import geopandas as gpd
import pandas as pd


@dataclass
class ColumnSpec:
    """Specification for a single column."""
    name: str
    dtype: Optional[str] = None
    nullable: bool = True
    unique: bool = False
    allowed_values: Optional[Set[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class Schema:
    """Schema specification for a DataFrame or GeoDataFrame."""
    name: str
    columns: List[ColumnSpec]
    required_columns: Optional[List[str]] = None
    row_count: Optional[int] = None
    min_rows: int = 0

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = [c.name for c in self.columns if not c.nullable]


class SchemaError(Exception):
    """Raised when schema validation fails."""
    pass


# Predefined Schemas
CD59_SCHEMA = Schema(
    name="cd59",
    columns=[
        ColumnSpec("boro_cd", dtype="Int64", nullable=False, unique=True),
        ColumnSpec("geometry", dtype="geometry", nullable=False),
    ],
    row_count=59,
)

DOMAIN_BASE_SCHEMA = Schema(
    name="domain_base",
    columns=[
        ColumnSpec("boro_cd", dtype="Int64", nullable=False, unique=True),
    ],
    min_rows=1,
)


def validate_column(df: pd.DataFrame, spec: ColumnSpec, context: str = "") -> List[str]:
    """Validate a single column against its specification."""
    errors = []
    col_name = spec.name

    if col_name not in df.columns:
        if col_name == "geometry" and isinstance(df, gpd.GeoDataFrame):
            pass
        else:
            errors.append(f"Missing column: {col_name}")
            return errors

    col = df[col_name]

    if spec.dtype is not None:
        if spec.dtype == "geometry":
            if not isinstance(df, gpd.GeoDataFrame):
                errors.append(f"Expected GeoDataFrame for geometry column {col_name}")
        elif spec.dtype == "Int64":
            if str(col.dtype) != "Int64":
                errors.append(f"Column {col_name}: expected Int64, got {col.dtype}")
        elif spec.dtype == "float64":
            if not pd.api.types.is_float_dtype(col):
                errors.append(f"Column {col_name}: expected float64, got {col.dtype}")

    if not spec.nullable and col.isna().any():
        errors.append(f"Column {col_name}: {col.isna().sum()} NA values not allowed")

    if spec.unique and col.duplicated().any():
        errors.append(f"Column {col_name}: {col.duplicated().sum()} duplicate values")

    if spec.allowed_values is not None:
        invalid = ~col.isin(spec.allowed_values) & col.notna()
        if invalid.any():
            errors.append(f"Column {col_name}: invalid values {list(col[invalid].unique()[:5])}")

    if spec.min_value is not None:
        if ((col < spec.min_value) & col.notna()).any():
            errors.append(f"Column {col_name}: values below min {spec.min_value}")

    if spec.max_value is not None:
        if ((col > spec.max_value) & col.notna()).any():
            errors.append(f"Column {col_name}: values above max {spec.max_value}")

    return errors


def validate_schema(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    schema: Schema,
    context: str = "",
    raise_on_error: bool = True,
) -> List[str]:
    """Validate a DataFrame against a schema."""
    errors = []
    ctx = f" ({context})" if context else ""

    if schema.row_count is not None and len(df) != schema.row_count:
        errors.append(f"Expected {schema.row_count} rows, got {len(df)}{ctx}")

    if len(df) < schema.min_rows:
        errors.append(f"Expected at least {schema.min_rows} rows, got {len(df)}{ctx}")

    missing = set(schema.required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}{ctx}")

    for col_spec in schema.columns:
        col_errors = validate_column(df, col_spec, context)
        errors.extend(col_errors)

    if errors and raise_on_error:
        raise SchemaError(
            f"Schema validation failed for '{schema.name}':\n" + "\n".join(errors)
        )

    return errors


def ensure_boro_cd_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure boro_cd column is Int64 dtype."""
    if "boro_cd" in df.columns:
        df = df.copy()
        df["boro_cd"] = df["boro_cd"].astype("Int64")
    return df


def validate_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, List[str]],
    how: str = "left",
    validate: str = "one_to_one",
    context: str = "",
) -> pd.DataFrame:
    """Perform a merge with validation."""
    try:
        return pd.merge(left, right, on=on, how=how, validate=validate)
    except pd.errors.MergeError as e:
        raise ValueError(f"Merge validation failed ({context}): {e}")


SCHEMAS: Dict[str, Schema] = {
    "cd59": CD59_SCHEMA,
    "domain_base": DOMAIN_BASE_SCHEMA,
}


def get_schema(name: str) -> Schema:
    """Get a registered schema by name."""
    if name not in SCHEMAS:
        raise KeyError(f"Unknown schema: {name}. Available: {list(SCHEMAS.keys())}")
    return SCHEMAS[name]


def register_schema(schema: Schema) -> None:
    """Register a new schema."""
    SCHEMAS[schema.name] = schema
