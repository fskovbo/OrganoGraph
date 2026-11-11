"""Fixed spherical-cap representations and opening-sensitivity experiments.

Core methods require NumPy/SciPy. VTK/libigl/Plotly are loaded only by mesh I/O,
optional geometric diagnostics, and organograph.plotting.crypt_morphology.
"""
from .alignment import (align_representations, align_to_reference, generalized_procrustes,
                        make_alignment_reference, weighted_procrustes)
from .basis import CapBasis
from .boundaries import (BoundaryFamily, ClippedPatch, OpeningShift, clip_level_set,
                         opening_shifts, prepare_boundary_family, restrict_mapping)
from .datatypes import (AlignmentReference, CapMapping, CapRepresentation,
                        ProcrustesResult, SimilarityTransform, SurfacePatch)
from .diagnostics import surface_distance_metrics
from .family_alignment import (SharedVertexPhase, align_known_boundary_variant,
                               reframe_representation, shared_vertex_phase)
from .decoding import (DecodingProbe, coefficient_surface_distance, decode_coefficients, interpolate_coefficients,
                       rim_coefficient_error, transport_coefficient_error)
from .mesh_quality import mesh_intersections, mesh_quality_metrics
from .examples import synthetic_crypt
from .io import (load_alignment_reference, load_basis, load_representation,
                 save_alignment_reference, save_basis, save_representation)
from .parameterization import parameterize_patch
from .patches import PatchTopologyError, extract_patch, surface_moments, vertex_areas
from .representation import fit_representation, reconstruct, with_size_mode
from .stability import StabilityResult, evaluate_opening_family

__all__ = [
    "CapBasis", "CapMapping", "CapRepresentation", "SurfacePatch", "SimilarityTransform",
    "AlignmentReference", "ProcrustesResult", "PatchTopologyError", "extract_patch",
    "parameterize_patch", "fit_representation", "reconstruct", "with_size_mode", "weighted_procrustes",
    "generalized_procrustes", "align_representations", "make_alignment_reference",
    "align_to_reference", "surface_distance_metrics", "surface_moments", "vertex_areas",
    "synthetic_crypt", "save_basis", "load_basis", "save_representation", "load_representation",
    "save_alignment_reference", "load_alignment_reference",
    "BoundaryFamily", "ClippedPatch", "OpeningShift", "clip_level_set", "opening_shifts",
    "prepare_boundary_family", "restrict_mapping", "StabilityResult", "evaluate_opening_family",
    "decode_coefficients", "interpolate_coefficients", "rim_coefficient_error",
    "transport_coefficient_error", "coefficient_surface_distance", "mesh_intersections", "mesh_quality_metrics",
    "DecodingProbe",
    "SharedVertexPhase", "align_known_boundary_variant", "reframe_representation", "shared_vertex_phase",
]
