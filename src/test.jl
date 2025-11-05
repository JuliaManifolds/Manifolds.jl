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
function test_exp end
function test_log end
function test_manifold_dimension end
end
