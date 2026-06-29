"""PDF report generation: annotate a document with its forensic findings."""

from .annotator import annotate_document, compute_fraud_risk

__all__ = ["annotate_document", "compute_fraud_risk"]
