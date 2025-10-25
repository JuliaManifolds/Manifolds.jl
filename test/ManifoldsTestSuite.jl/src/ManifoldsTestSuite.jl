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
# --- E
"""
    test_exp_log(G::AbstractManifold, p, q, X; kwargs...)

Test  `exp` and `log` for given points `p, q` and a tangent vector `X` at `p`.

!!! note
    This function requires `is_point(M, p)` and `is_vector(M, p, X))` to be implemented

# Keyword arguments

* `atol::Real=0`: the absolute tolerance for the tests of zero-vectors
* `test_exp::Bool=true`: test the exponential map yields a point on `G`
* `test_log::Bool=true`: test the logarithmic map.
* `test_mutating::Bool=true`: test the mutating functions
* `test_fused::Bool=true` test the `exp_fused` function
"""
function test_exp_log(
        M::AbstractManifold,
        p,
        q,
        X;
        atol::Real = 0,
        test_exp::Bool = true,
        test_mutating::Bool = true,
        test_log::Bool = true,
        test_fused::Bool = true,
    )
    @testset "exp & log" begin
        if test_exp
            # Lie group exp
            p1 = exp(G, p, X)
            if test_mutating
                p2 = copy(M, p)
                exp!(M, p2, p, X)
                @test isapprox(G, p1, p2)
            end
            @test is_point(G, p1; error = :error)
            if test_fused
                p3 = Manifolds.exp_fused(M, p, X, 1.0)
                if test_mutating
                    p4 = copy(M, p)
                    Manifolds.exp_fused!(M, p4, p, X, 1.0)
                    @test isapprox(G, p3, p4)
                end
                @test is_point(G, p3; error = :error)
            end
        end
        if test_log
            Y1 = log(M, p, q)
            @test is_vector(M, p, Y1; error = :error)
            @test isapprox(M, p, zero_vector(M, p), log(G, p, p); atol = atol)
            @test isapprox(M, q, zero_vector(M, q), log(G, q, q); atol = atol)
            if test_mutating
                Y2 = zero_vector(M, p)
                log!(G, Y2, p, q)
                @test isapprox(M, p, Y1, Y2)
                log!(G, Y2, p, p)
                @test isapprox(M, p, Y2, zero_vector(M, p))
                log!(G, Y2, q, q)
                @test isapprox(M, q, Y2, zero_vector(M, q))
            end
        end
        if test_exp && test_log
            p3 = exp(G, p, X)
            Y4 = log(G, p, p3)
            @test isapprox(M, p, X, Y4)
            if test_mutating
                p4 = copy(M, p)
                Y5 = zero_vector(M, p)
                log!(M, Y5, p, exp!(G, p4, p, X))
                @test isapprox(M, p, Y4, Y5)
            end
        end
    end
    return nothing
end

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
        dimension::Union{<:Integer, Missing} = missing,
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
# --- R
"""
    test_representation_size(M::AbstractManifold, size)

Test that the `representation_size` method works as expected.

`size` can be an integer or a tuple of integers, e.g. for vectors and matrices,
respectively. It can be `nothing` if the manfold has a repesentation where a size does not make sense.
If the `size` is `missing` there is no actual test performed, but just whether it does not error
"""
function test_representation_size(M::AbstractManifold, size)
    @testset "representation_size(M)" begin
        rs = representation_size(M)
        if !ismissing(size)
            if size === nothing
                @test isnothing(rs)
            else
                @test rs == size
            end
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


#
# Main test method
# ------------------------------------------------------------------------------------------

"""
    test_manifold(G::AbstractManifold, properties::Dict, expectations::Dict)

Test a manifold `M`based on a `Dict` of `properties` what to test
and a `Dict` of `expectations` containing values to expect as solutions.

Possible properties are

* `:Functions` a vector of functions to test
* `:Points` a vector of points for the tests, depending on which function needs them
* `:Vectors` a vector of tangent vectors, each in the tangent space of the corresponding entry of `:Points`,
  depending on the functions you test
* `:NonPoints` a vector of “things” that are not points on the manifold, used within `is_point`
* `:NonVectors` a vector 2-tuples ``(p,V)` of (non-)points and non-vectors used within is_vector
* `:Mutating` is a boolean (`true` by default) whether to test the mutating variants of functions or not.
* `:Name` is a name of the test. If not provided, defaults to `"\$M"`
* `:Rng` is a random number generator, if provided, the random functions are tested with this generator as well
* `:RetractionMethods` is a vector of retraction methods to test, if `retract` is in `:Functions`
* `:InverseRetractionMethods` is a vector of inverse retraction methods to test, if `inverse_retract` is in `:Functions`
* `:VectorTransportMethods` is a vector of vector transport methods to test,
  if `vector_transport_to` or `vector_transport_direction` is in `:Functions`

for each function `f`, there is also always the single function `test_f(M, args; kwargs...)`,

Possible `expectations` are

* `:atol => 0.0` a global absolute tolerance
* `:atols -> Dict()` a dictionary `function -> atol` for specific function tested.
* `:repr` is a sting one gets from `repr(M)` or when calling `show`
"""
function test_manifold(M::AbstractManifold, properties::Dict, expectations = Dict())
    atol = get(expectations, :atol, 0.0)
    function_atols = get(expectations, :atols, Dict())
    functions = get(properties, :Functions, Function[])
    mutating = get(properties, :Mutating, true)
    points = get(properties, :Points, [])
    retraction_methods = get(properties, :RetractionMethods, AbstractRetractionMethod[])
    inverse_retraction_methods = get(properties, :InverseRetractionMethods, AbstractInverseRetractionMethod[])
    vectortransport_methods = get(properties, :VectorTransportMethods, AbstractVectorTransportMethod[])
    nonpoints = get(properties, :NonPoints, [])
    nonvectors = get(properties, :NonVectors, [])
    vectors = get(properties, :Vectors, [])
    test_name = get(properties, :Name, "$M")
    return @testset "$(test_name)" begin
        #
        #
        # --- E
        if any(in.([exp, log], Ref(functions)))
            exp_atol = get(function_atols, exp, atol)
            log_atol = get(function_atols, log, atol)
            local_atol = max(exp_atol, log_atol, atol)
            (length(points) < 2) && error("Testing exp/log requires at least 2 points, only $(length(points)) provided.")
            (length(vectors) < 1) && error("Testing exp/log requires at least one tangent vector, but non was provided.")
            test_exp_log(
                M, points[1], points[2], vectors[1];
                atol = local_atol,
                test_exp = (exp in functions),
                test_log = (log in functions),
                test_mutating = mutating,
            )
        end
        #
        #
        # --- M
        if (manifold_dimension ∈ functions)
            v = get(expectations, :manifold_dimension, missing)
            test_manifold_dimension(M, v)
        end
        #
        #
        # --- R
        if (representation_size ∈ functions)
            v = get(expectations, :representation_size, missing)
            test_representation_size(M, v)
        end
        #
        #
        # --- S
        if (any(in.([show, repr], Ref(functions)))) && haskey(expectations, :repr)
            test_show(M, expectations[:repr])
        end
    end
end
end
