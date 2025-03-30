"""
    ManifoldsTestSuite.jl

An internal small module to encapsulate

* dummy types
* test functions
* a test suite

This is a slow work-in-progress to replace the `test_manifold` function,
and encapsulate and collect common tools for testing a manifold.

The current plan is to slowly move parts to this test suite whenever revising some tests or¨
time permits.
"""
module ManifoldsTestSuite
using Manifolds
using Test, Random

#
#
# --- M
"""
    test_manifold_dimension(M::AbstractManifold, dimension::Union{Int,Missing}=missing)

Test the manifold dimension
* to be non-negative
* to be an integer
* to be the expected `dimension` if provided.

Requires `manifold_dimension` to be implemented.
"""
function test_manifold_dimension(
    M::AbstractManifold,
    dimension::Union{<:Integer,Missing}=missing,
)
    @testset "manifold_dimension(M)" begin
        d = manifold_dimension(M)
        @test d ≥ 0
        @test isinteger(d)
        if !ismissing(dimension)
            @test d == dimension
        end
    end
    return nothing
end

#
#
# --- S
"""
    test_show(M::AbstractManifold, repr_string::AbstractString)

Test that show methods work as expected.
For now this (only) checks that `"\$G"` yields the `repr_string`.

Requires `show` (or `repr`) to be implemented.
"""
function test_show(M::AbstractManifold, repr_string::AbstractString)
    @testset "repr(M)" begin
        @test repr(M) == repr_string
    end
    return nothing
end

"""
    test_manifold(G::AbstractManifold, properties::Dict, expectations::Dict)

Test a manifold `M`based on a `Dict` of `properties` what to test
and a `Dict` of `expectations` containing values to expect as solutions.

Possible properties are

* `:Functions` a vector of functions to test
* `:Points` a vector of points for the tests, depending on which function needs them
* `:Vectors` a vector of tangent vectors, each in the tangent space of the corresponding entry of `:Points`,
  depending on the functions you test
* `:Mutating` is a boolean (`true` by default) whether to test the mutating variants of functions or not.
* `:Name` is a name of the test. If not provided, defaults to `"\$G"`
* `:Rng` is a random number generator, if provided, the random functions are tested with this generator as well

for each function `f`, there is also always the single function `test_f(M, args; kwargs...)`,


Possible `expectations` are

* `:atol => 0.0` a global absolute tolerance
* `:atols -> Dict()` a dictionary `function -> atol` for specific function tested.
* `:repr` is a sting one gets from `repr(M)` or when calling `show`
"""
function test_manifold(M::AbstractManifold, properties::Dict, expectations=Dict())
    atol = get(expectations, :atol, 0.0)
    mutating = get(properties, :Mutating, true)
    functions = get(properties, :Functions, Function[])
    points = get(properties, :Points, [])
    vectors = get(properties, :Vectors, [])
    test_name = get(properties, :Name, "$G")
    function_atols = get(expectations, :atols, Dict())
    @testset "$(test_name)" begin
        if (manifold_dimension ∈ functions)
            v = get(expectations, :manifold_dimension, missing)
            test_manifold_dimension(M, v)
        end
        if (any(in.([show, repr], Ref(functions)))) && haskey(expectations, :repr)
            test_show(M, expectations[:repr])
        end
    end
end
end
