"""Numerical primitives for the entropy and dyadic-sampling implementations."""
from types import SimpleNamespace

from .numerics import directional_observations, convex_profile_replay
from .numerics.directional_observations import _point
from .numerics.metadata_entropy_audit import control_observation
from .numerics.quad_profile_convex import ConvexEntropyProfile
from .numerics.direct_convex_numeric import certify
from .numerics.ppr_convex import _I, _D, _fdown, _fup
from .numerics.certified_entropy import _exp
from .numerics.directional_numeric import wealth_lower

DEFAULT_REFERENCE_ROOT = None


def _load_reference_modules(reference_root=None):
    return {
        'directional_observations': directional_observations,
        'convex_profile_replay': convex_profile_replay,
        'convex_audit_online': SimpleNamespace(
            _point=_point, control_observation=control_observation,
            ConvexEntropyProfile=ConvexEntropyProfile, certify=certify,
            _I=_I, _D=_D, _exp=_exp, _fdown=_fdown, _fup=_fup,
            wealth_lower=wealth_lower),
    }
