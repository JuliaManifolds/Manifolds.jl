"""
    Manifolds.Test

The module `Manifolds.Test` contains functions to test functions from `Manifolds.jl`.
Most functionality is provided only when `Test.jl` is loaded as well, i.e. populated
with methods by the extension.

The test functions provided are mean to verify that the defined functions
on a specific manifold behave as expected, for example that
an allocating and in-place version of a function give the same result,
or that for example the exponential map computes a point on the manifold,
and calling the logarithmic map on the result yields the original tangent vector again,
for tangent vectors within the injectivity radius.

In general for every function defined in the API for manifolds,
this module shall provide a test function with
* the same name prefixed by `test_`
* the same function arguments as the original function
* keyword arguments to control which tests to perform
"""
module Test
using ..Manifolds
const _ALL_FUNCTIONS = [
    copy,
    copyto!,
    default_inverse_retraction_method,
    default_retraction_method,
    default_vector_transport_method,
    distance,
    exp,
    injectivity_radius,
    inner,
    inverse_retract,
    is_point,
    is_vector,
    log,
    manifold_dimension,
    norm,
    parallel_transport_direction,
    parallel_transport_to,
    retract,
    vector_transport_direction,
    vector_transport_to,
    zero_vector,
]
all_functions() = _ALL_FUNCTIONS
#
#
# the overall global interface
function test_manifold end
#
#
# the small functions per single API function
function test_copy end
function test_copyto end
function test_default_retraction end
function test_default_inverse_retraction end
function test_default_vector_transport_method end
function test_distance end
function test_exp end
function test_injectivity_radius end
function test_inner end
function test_inverse_retract end
function test_is_point end
function test_is_vector end
function test_log end
function test_manifold_dimension end
function test_norm end
function test_parallel_transport end
function test_repr end
function test_retract end
function test_vector_transport end
function test_zero_vector end
#
#
# TODO
function test_embed end
function test_estimated_sectional_curvature_matrix end
function test_flat end
function test_geodesic end
function test_get_basis end
function test_get_coordinates end
function test_get_embedding end
function test_get_vector end
function test_get_vectors end
function test_manifold_volume end
function test_mid_point end
function test_project end
function test_project_embed end
function test_rand end
function test_sectional_curvature_matrix end
function test_sharp end
function test_shortest_geodesic end
function test_Weingarten end
end
