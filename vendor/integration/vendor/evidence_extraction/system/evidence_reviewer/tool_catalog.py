from __future__ import annotations


# Suggestion-only catalog. Workflow structured never imports or executes these tools.
TOOL_CATALOG: list[dict[str, object]] = [
    {
        "name": "retrieve_source_context",
        "purpose": "Retrieve nearby original text when the supplied quote does not establish a field's meaning.",
        "required_inputs": ["article_id", "query or source quote"],
    },
    {
        "name": "derive_complement_count",
        "purpose": "Derive target = total - complement only after the source establishes an exhaustive two-part partition.",
        "required_inputs": [
            "total_count",
            "observed_complement_count",
            "source evidence for mutually exclusive and exhaustive categories",
        ],
    },
    {
        "name": "calculate_proportion",
        "purpose": "Calculate a proportion from a source-supported numerator and denominator.",
        "required_inputs": ["numerator", "denominator", "meaning of both counts"],
    },
    {
        "name": "calculate_binary_effect",
        "purpose": "Calculate RR or OR from source-supported two-arm event counts and denominators.",
        "required_inputs": [
            "intervention_events",
            "intervention_total",
            "control_events",
            "control_total",
            "requested effect type",
        ],
    },
    {
        "name": "validate_effect_interval",
        "purpose": "Check effect and confidence-interval ordering and consistency.",
        "required_inputs": ["effect value", "CI lower", "CI upper", "effect family"],
    },
]
