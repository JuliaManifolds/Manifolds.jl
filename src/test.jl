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

"""
    AbstractExpectation

An abstract type for expectations in tests.
"""
abstract type AbstractExpectation end

"""
    Expect{T} <: AbstractExpectation

A struct to hold expected values for tests of type `T`.

# Fields
*  `value::T` the expected value

# Constructor

    Expect(value::T)
"""
struct Expect{T} <: AbstractExpectation
    value::T
end
Expect(e::Expect) = e

"""
    NoExcpectation <: AbstractExpectation

A struct to indicate that no expectation is provided for a test.
"""
struct NoExpectation <: AbstractExpectation end

Expect(ne::NoExpectation) = ne

isexpected(e::Expect) = true
isexpected(value) = true
isexpected(e::NoExpectation) = false
"""
    expected(e::Union{Expect, NoExpectation})

Check if an expectation is provided.
"""
isexpected(e::Union{Expect, NoExpectation})

expect(e::Expect{T}) where {T} = e.value
expect(value) = value
expect(e::NoExpectation) = error("No expectation provided.")
"""
    expect(e::Union{Expect{T}, NoExpectation}) where T

Get the expected value if provided, error otherwise.
"""
expect(e::Union{Expect, NoExpectation})

#
#
# Dummy Types
struct DummyMatrixType <: Manifolds.AbstractMatrixType end

const _ALL_FUNCTIONS = [
    copy,
    copyto!,
    default_inverse_retraction_method,
    default_retraction_method,
    default_vector_transport_method,
    distance,
    embed,
    embed_project,
    exp,
    flat,
    geodesic,
    get_basis,
    get_coordinates,
    get_embedding,
    get_vector,
    get_vectors,
    injectivity_radius,
    inner,
    inverse_retract,
    is_default_metric,
    is_flat,
    is_point,
    is_vector,
    log,
    manifold_dimension,
    manifold_volume,
    mid_point,
    norm,
    parallel_transport_direction,
    parallel_transport_to,
    project,
    rand,
    repr,
    representation_size,
    retract,
    sectional_curvature,
    sharp,
    shortest_geodesic,
    vector_transport_direction,
    vector_transport_to,
    volume_density,
    Weingarten,
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
function test_embed end
function test_embed_project end
function test_exp end
function test_flat end
function test_geodesic end
function test_get_basis end
function test_get_coordinates end
function test_get_embedding end
function test_get_vector end
function test_get_vectors end
function test_injectivity_radius end
function test_inner end
function test_inverse_retract end
function test_is_default_metric end
function test_is_flat end
function test_is_point end
function test_is_vector end
function test_log end
function test_manifold_dimension end
function test_manifold_volume end
function test_mid_point end
function test_norm end
function test_parallel_transport end
function test_project end
function test_sectional_curvature end
function test_sharp end
function test_shortest_geodesic end
function test_rand end
function test_repr end
function test_representation_size end
function test_retract end
function test_vector_transport end
function test_volume_density end
function test_Weingarten end
function test_zero_vector end
#
#
# For now not yet part of the test suite, mainly because testing them is not that easy to do
# generically.
# Please test these individually
# How to test is is approx ok?
function test_estimated_sectional_curvature_matrix end
# What to exactly test on this?
function test_sectional_curvature_matrix end
# How approximate is ok?
function test_mean end
function test_median end
end
