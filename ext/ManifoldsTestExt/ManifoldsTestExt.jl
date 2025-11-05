module ManifoldsTestExt

using Manifolds
using ManifoldsBase
using Test

"""
    text_exp(
        M, p, X;
        available_functions=[], expected_value=nothing, test_mutating=true,
        test_log = (log in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        kwargs...
    )
    Test the exponential map on manifold `M` at point `p` with tangent vector `X`.

    * that the result is a valid point on the manifold
    * that the result matches `expected_value`, if given
    * that the mutating version `exp!` matches the non-mutating version, (if activated)
    * that the logarithmic map inverts the exponential map (if activated)
      (only performed if either `injectivity_radius` is not available or `X` is within)
"""
function Manifolds.Test.test_exp(
        M::AbstractManifold, p, X;
        available_functions = Function[],
        expected_value = nothing,
        test_log = (log in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        test_mutating = true,
        kwargs...
    )
    @testset "Exponential map" begin
        q = exp(M, p, X)
        @test is_point(M, q; error = :error, kwargs...)
        if !isnothing(expected_value)
            @test isapprox(M, q, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            q2 = copy(M, p)
            exp!(M, q2, p, X)
            @test isapprox(M, q2, q; error = :error, kwargs...)
        end
        if test_log
            # Test only if inj is not available of X is within inj radius
            run_test = !test_injectivity_radius || norm(M, p, X) ≤ injectivity_radius(M, p)
            run_test || (@warn("Skipping log-exp test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"); nothing)
            Y = log(M, p, q)
            @test isapprox(M, X, Y; error = :error, kwargs...) skip = !run_test
        end
    end
    return nothing
end # Manifolds.Test.test_exp
"""
    text_log(
        M, p, q;
        available_functions=[], expected_value=nothing, test_mutating=true,
        test_exp = (exp in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        kwargs...
    )
    Test the logarithmic map on manifold `M` at point `p` towards q

    * that the result is a valid tangent vector at `p` on the manifold
    * that the result matches `expected_value`, if given
    * that the mutating version `log!` matches the non-mutating version, (if activated)
    * that the exponential map inverts the logarithmic map (if activated)
    (only performed if either `injectivity_radius` is not available or `X` is within)
"""
function Manifolds.Test.test_log(
        M::AbstractManifold, p, q;
        available_functions = Function[],
        expected_value = nothing,
        test_exp = (exp in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        test_mutating = true,
        kwargs...
    )
    @testset "Logarithmic map" begin
        X = log(M, p, q)
        @test is_vector(M, p, X; error = :error, kwargs...)
        if !isnothing(expected_value)
            @test isapprox(M, p, X, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            Y = copy(M, p, X)
            log!(M, Y, p, q)
            @test isapprox(M, Y, X; error = :error, kwargs...)
        end
        if test_exp
            # Test only if inj is not available of X is within inj radius
            run_test = !test_injectivity_radius || norm(M, p, X) ≤ injectivity_radius(M, p)
            run_test || (@warn("Skipping exp-log test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"); nothing)
            q2 = exp(M, p, X)
            @test isapprox(M, q2, q; error = :error, kwargs...) skip = !run_test
        end
    end
    return nothing
end # Manifolds.Test.test_log
"""
    test_manifold(G::AbstractManifold, properties::Dict, expectations::Dict)

Test the [`AbstractManifold`](@ref) ``\\mathcal M`` based on a `Dict` of properties and a `Dict` of `expectations`.

Possible properties are

* `:Aliased` is a boolean (same as `:Mutating` by default) whether to test the mutating variants with aliased input
* `:Functions` is a vector of all defined functions for `M`
  note a test is activated by the function (like `exp`), adding the mutating function (like `exp!`) overwrites the
  global default (see `:Mutating`) to true.
* `:Points` is a vector of at least 2 points on `M`, which should not be the same point
* `:Vectors` is a vector of at least 2 tangent vectors, which should be in the tangent space of the correspondinig point entr in `:Points`
* `:Mutating` is a boolean (`true` by default) whether to test the mutating variants of functions or not.
* `:Name` is a name of the test. If not provided, defaults to `"\$M"`

Possible entries of the `expectations` dictionary are

* `exp` for the result of `exp(M, p, X)`
* `log` for the result of `log(M, p, q)`
* `:atol => 0.0` a global absolute tolerance
* `:atols -> Dict()` a dictionary `function -> atol` for specific function tested.
"""
function Manifolds.Test.test_manifold(G::AbstractManifold, properties::Dict, expectations::Dict = Dict())
    atol = get(expectations, :atol, 0.0)
    mutating = get(properties, :Mutating, true)
    aliased = get(properties, :Aliased, mutating)
    functions = get(properties, :Functions, Function[])
    points = get(properties, :Points, [])
    vectors = get(properties, :Vectors, [])
    test_name = get(properties, :Name, "$G")
    function_atols = get(expectations, :atols, Dict())
    return @testset "$test_name" begin
        n_points = length(points)
        n_vectors = length(vectors)
        if (exp in functions)
            p = points[1]
            X = vectors[1]
            expected_exp = get(expectations, exp, nothing)
            Manifolds.Test.test_exp(
                G, p, X;
                available_functions = functions,
                expected_value = expected_exp,
                test_mutating = (exp! in functions) ? true : mutating,
                atol = get(function_atols, exp, atol),
            )
        end
        if (log in functions)
            p = points[1]
            q = points[2]
            expected_log = get(expectations, :log, nothing)
            Manifolds.Test.test_log(
                G, p, q;
                available_functions = functions,
                expected_value = expected_log,
                test_mutating = (log! in functions) ? true : mutating,
                atol = get(function_atols, log, atol),
            )
        end
    end

end
# old includes
include("tests_general.jl")

end
