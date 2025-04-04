"""
JSON Schema definition for pytics profile output validation
"""

PYTICS_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["metadata", "overview", "variables"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["title", "generated_at", "pytics_version", "schema_version"],
            "properties": {
                "title": {"type": "string"},
                "generated_at": {
                    "type": "string",
                    "format": "date-time"
                },
                "pytics_version": {"type": "string"},
                "schema_version": {"type": "string"}
            }
        },
        "overview": {
            "type": "object",
            "required": [
                "shape", "n_vars", "n_obs", "n_missing", "missing_percent",
                "n_duplicates", "duplicates_percent", "memory_usage", "avg_record_size"
            ],
            "properties": {
                "shape": {
                    "type": "object",
                    "required": ["rows", "columns"],
                    "properties": {
                        "rows": {"type": "integer", "minimum": 0},
                        "columns": {"type": "integer", "minimum": 0}
                    }
                },
                "n_vars": {"type": "integer", "minimum": 0},
                "n_obs": {"type": "integer", "minimum": 0},
                "n_missing": {"type": "integer", "minimum": 0},
                "missing_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "n_duplicates": {"type": "integer", "minimum": 0},
                "duplicates_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "n_numeric": {"type": "integer", "minimum": 0},
                "n_categorical": {"type": "integer", "minimum": 0},
                "n_boolean": {"type": "integer", "minimum": 0},
                "n_date": {"type": "integer", "minimum": 0},
                "n_text": {"type": "integer", "minimum": 0},
                "memory_usage": {"type": "string"},
                "avg_record_size": {"type": "string"}
            }
        },
        "index_analysis": {
            "type": "object",
            "required": [
                "name", "dtype", "unique_count", "is_monotonic_increasing",
                "is_monotonic_decreasing", "has_duplicates", "memory_usage"
            ],
            "properties": {
                "name": {"type": "string"},
                "dtype": {"type": "string"},
                "unique_count": {"type": "integer", "minimum": 0},
                "is_monotonic_increasing": {"type": "boolean"},
                "is_monotonic_decreasing": {"type": "boolean"},
                "has_duplicates": {"type": "boolean"},
                "memory_usage": {"type": ["integer", "number"]}
            }
        },
        "variables": {
            "type": "object",
            "patternProperties": {
                "^.*$": {  # Match any variable name
                    "type": "object",
                    "required": [
                        "type", "missing_count", "missing_percent", "distinct_count",
                        "distinct_percent", "memory_usage", "statistics", "distribution"
                    ],
                    "properties": {
                        "type": {"type": "string"},
                        "missing_count": {"type": "integer", "minimum": 0},
                        "missing_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "distinct_count": {"type": "integer", "minimum": 0},
                        "distinct_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "memory_usage": {"type": "integer", "minimum": 0},
                        "statistics": {
                            "type": "object",
                            "oneOf": [
                                {  # Numeric statistics
                                    "required": [
                                        "mean", "std", "min", "max", "median", "q1", "q3",
                                        "sum", "skewness", "kurtosis"
                                    ],
                                    "properties": {
                                        "mean": {"type": ["number", "string"]},  # string for "Infinity"
                                        "std": {"type": "number"},
                                        "min": {"type": ["number", "string"]},
                                        "max": {"type": ["number", "string"]},
                                        "median": {"type": "number"},
                                        "q1": {"type": "number"},
                                        "q3": {"type": "number"},
                                        "sum": {"type": ["number", "string"]},
                                        "skewness": {"type": "number"},
                                        "kurtosis": {"type": "number"}
                                    }
                                },
                                {  # Boolean statistics
                                    "required": [
                                        "true_count", "false_count",
                                        "true_percent", "false_percent"
                                    ],
                                    "properties": {
                                        "true_count": {"type": "integer", "minimum": 0},
                                        "false_count": {"type": "integer", "minimum": 0},
                                        "true_percent": {"type": "number", "minimum": 0, "maximum": 100},
                                        "false_percent": {"type": "number", "minimum": 0, "maximum": 100}
                                    }
                                },
                                {  # Datetime statistics
                                    "required": ["min_date", "max_date", "range"],
                                    "properties": {
                                        "min_date": {"type": "string", "format": "date-time"},
                                        "max_date": {"type": "string", "format": "date-time"},
                                        "range": {"type": "string"}
                                    }
                                },
                                {  # Categorical statistics
                                    "required": [
                                        "distinct_count", "top_frequent_value", "frequency"
                                    ],
                                    "properties": {
                                        "distinct_count": {"type": "integer", "minimum": 0},
                                        "top_frequent_value": {"type": "string"},
                                        "frequency": {"type": "integer", "minimum": 0}
                                    }
                                }
                            ]
                        },
                        "distribution": {
                            "type": "object",
                            "oneOf": [
                                {  # Histogram distribution
                                    "required": ["type", "counts", "bin_edges"],
                                    "properties": {
                                        "type": {"const": "histogram"},
                                        "counts": {
                                            "type": "array",
                                            "items": {"type": "number", "minimum": 0}
                                        },
                                        "bin_edges": {
                                            "type": "array",
                                            "items": {"type": "number"}
                                        }
                                    }
                                },
                                {  # Boolean distribution
                                    "required": ["type", "counts"],
                                    "properties": {
                                        "type": {"const": "boolean"},
                                        "counts": {
                                            "type": "object",
                                            "properties": {
                                                "true": {"type": "integer", "minimum": 0},
                                                "false": {"type": "integer", "minimum": 0}
                                            }
                                        }
                                    }
                                },
                                {  # Datetime distribution
                                    "required": ["type", "by_month"],
                                    "properties": {
                                        "type": {"const": "datetime"},
                                        "by_month": {
                                            "type": "object",
                                            "required": ["periods", "counts"],
                                            "properties": {
                                                "periods": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                },
                                                "counts": {
                                                    "type": "array",
                                                    "items": {"type": "integer", "minimum": 0}
                                                }
                                            }
                                        }
                                    }
                                },
                                {  # Categorical distribution
                                    "required": ["type", "categories", "counts"],
                                    "properties": {
                                        "type": {"const": "categorical"},
                                        "categories": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "counts": {
                                            "type": "array",
                                            "items": {"type": "integer", "minimum": 0}
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        },
        "correlations": {
            "type": "object",
            "required": ["matrix", "columns"],
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -1, "maximum": 1}
                    }
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "missing_values": {
            "type": "object",
            "required": ["by_column", "patterns"],
            "properties": {
                "by_column": {
                    "type": "object",
                    "patternProperties": {
                        "^.*$": {"type": "integer", "minimum": 0}
                    }
                },
                "patterns": {
                    "type": "object",
                    "patternProperties": {
                        "^[0-9]+$": {"type": "integer", "minimum": 0}
                    }
                }
            }
        },
        "duplicates": {
            "type": "object",
            "required": ["count", "percent", "examples"],
            "properties": {
                "count": {"type": "integer", "minimum": 0},
                "percent": {"type": "number", "minimum": 0, "maximum": 100},
                "examples": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            }
        },
        "target_analysis": {
            "type": "object",
            "required": ["statistics", "distribution"],
            "properties": {
                "statistics": {"type": "object"},
                "distribution": {"type": "object"}
            }
        }
    }
} 