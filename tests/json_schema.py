"""JSON schema for pytics profile output validation."""

PYTICS_JSON_SCHEMA = {
    "type": "object",
    "required": ["metadata", "overview", "variables"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["title", "generated_at", "pytics_version", "schema_version"],
            "properties": {
                "title": {"type": "string"},
                "generated_at": {"type": "string", "format": "date-time"},
                "pytics_version": {"type": "string"},
                "schema_version": {"type": "string"}
            }
        },
        "overview": {
            "type": "object",
            "required": ["shape", "n_vars", "n_obs", "memory_usage"],
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
                "memory_usage": {"type": "string"}
            }
        },
        "variables": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "required": ["type", "missing_count"],
                    "properties": {
                        "type": {"type": "string"},
                        "missing_count": {"type": "integer", "minimum": 0},
                        "statistics": {
                            "type": "object",
                            "properties": {
                                "mean": {"type": ["number", "null"]},
                                "std": {"type": ["number", "null"]},
                                "min": {"type": ["number", "null"]},
                                "25%": {"type": ["number", "null"]},
                                "50%": {"type": ["number", "null"]},
                                "75%": {"type": ["number", "null"]},
                                "max": {"type": ["number", "null"]},
                                "mode": {"type": ["string", "number", "null"]},
                                "unique_count": {"type": "integer", "minimum": 0}
                            }
                        },
                        "distribution": {
                            "type": "object",
                            "required": ["type"],
                            "properties": {
                                "type": {"type": "string", "enum": ["histogram", "bar", "none"]},
                                "counts": {"type": "array", "items": {"type": "number"}},
                                "bin_edges": {"type": "array", "items": {"type": "number"}},
                                "categories": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                }
            }
        },
        "correlations": {
            "type": "object",
            "properties": {
                "pearson": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {
                            "type": "object",
                            "patternProperties": {
                                ".*": {"type": ["number", "null"]}
                            }
                        }
                    }
                }
            }
        },
        "missing_values": {
            "type": "object",
            "properties": {
                "total_missing": {"type": "integer", "minimum": 0},
                "missing_percentage": {"type": "number", "minimum": 0, "maximum": 100},
                "variables_with_missing": {"type": "integer", "minimum": 0}
            }
        },
        "duplicates": {
            "type": "object",
            "properties": {
                "total_duplicates": {"type": "integer", "minimum": 0},
                "duplicate_percentage": {"type": "number", "minimum": 0, "maximum": 100}
            }
        },
        "target_analysis": {
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "target_type": {"type": "string"},
                "feature_importance": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            }
        }
    }
} 