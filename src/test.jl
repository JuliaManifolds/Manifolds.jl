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
"""
module Test
#
#
# the overall global interface
function test_manifold end
#
#
# the small functions per single API function
function test_exp end # TODO: Test fused?
function test_log end
function test_manifold_dimension end
#
#
# TODO
function test_copy end
function test_copyto end
function test_default_atlas end
function test_default_retraction end
function test_default_inverse_retraction end
function test_default_vector_transport end
function test_distance end
function test_embed end
function test_flat end
function test_geodesic end
function test_get_basis end
function test_get_coordinates end
function test_get_vector end
function test_get_vectors end
function test_injectivity_radius end
function test_inner end
function test_inverse_retraction end
function test_is_point end
function test_is_vector end
function test_mid_point end
function test_norm end
function test_parallel_transport end
function test_project end
function test_project_embed end
function test_rand end
function test_repr end
function test_retraction end
function test_sharp end
function test_shortest_geodesic end
function test_vector_transport end
function test_zero_vector end
end
