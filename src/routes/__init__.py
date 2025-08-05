"""
Routes package for FPSO equipment monitoring.

This package contains orchestration logic for different workflows and pipelines.
"""

from .data_processing import DataProcessingRoutes

__all__ = ['DataProcessingRoutes']
