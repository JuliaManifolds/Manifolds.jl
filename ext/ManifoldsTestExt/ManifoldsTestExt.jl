module ManifoldsTestExt

using Manifolds
using ManifoldsBase
using Test

#
#
#
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
* `manifold_dimension` for the expected dimension of the manifold
* `:atol => 0.0` a global absolute tolerance
* `:atols -> Dict()` a dictionary `function -> atol` for tolerances of specific function tested.
* `:Types` -> Dict() a dictionary `function -> Type` for specifying expected types of results of specific functions.

"""
function Manifolds.Test.test_manifold(M::AbstractManifold, properties::Dict, expectations::Dict = Dict())
    atol = get(expectations, :atol, 0.0)
    mutating = get(properties, :Mutating, true)
    aliased = get(properties, :Aliased, mutating)
    functions = get(properties, :Functions, Function[])
    points = get(properties, :Points, [])
    vectors = get(properties, :Vectors, [])
    test_name = get(properties, :Name, "$M")
    function_atols = get(expectations, :atols, Dict())
    result_types = get(expectations, :Types, Dict())
    return @testset "$test_name" begin
        n_points = length(points)
        n_vectors = length(vectors)
        if (exp in functions)
            Manifolds.Test.test_exp(
                M, points[1], vectors[1];
                available_functions = functions,
                expected_value = get(expectations, exp, nothing),
                test_aliased = aliased,
                test_mutating = (exp! in functions) ? true : mutating,
                atol = get(function_atols, exp, atol),
                name = "exp(M, p, X)", # shorten name within large suite
            )
        end
        if (log in functions)
            expected_log = get(expectations, :log, nothing)
            Manifolds.Test.test_log(
                M, points[1], points[2];
                available_functions = functions,
                expected_value = expected_log,
                test_mutating = (log! in functions) ? true : mutating,
                atol = get(function_atols, log, atol),
                name = "log(M, p, q)", # shorten name within large suite
            )
        end
        if (manifold_dimension in functions)
            expected_dim = get(expectations, manifold_dimension, nothing)
            Manifolds.Test.test_manifold_dimension(
                M; expected_value = expected_dim, expected_type = get(result_types, manifold_dimension, Int),
                name = "manifold_dimension(M)",
            )
        end
    end
end

# Single function tests
#
# ------------------------------------------------------------------------------------------
"""
    test_exp(
        M, p, X;
        available_functions=[], expected_value=nothing, test_mutating=true,
        test_log = (log in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        name = "Exponential map on\$M for \$(typeof(p)) points",
        kwargs...
    )
    Test the exponential map on manifold `M` at point `p` with tangent vector `X`.

    * that the result is a valid point on the manifold
    * that the result matches `expected_value`, if given
    * that the mutating version `exp!` matches the non-mutating version, (if activated)
    * that `exp!` works on aliased in put (`p=q`) (if activated for mutating)
    * that the logarithmic map inverts the exponential map (if activated)
      (only performed if either `injectivity_radius` is not available or `X` is within)
"""
function Manifolds.Test.test_exp(
        M::AbstractManifold, p, X;
        available_functions = Function[],
        expected_value = nothing,
        test_aliased = true,
        test_log = (log in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        test_mutating = true,
        name = "Exponential map on $M for $(typeof(p)) points",
        kwargs...
    )
    @testset "$(name)" begin
        q = exp(M, p, X)
        @test is_point(M, q; error = :error, kwargs...)
        if !isnothing(expected_value)
            @test isapprox(M, q, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            q2 = copy(M, p)
            exp!(M, q2, p, X)
            @test isapprox(M, q2, q; error = :error, kwargs...)
            if test_aliased
                q3 = copy(M, p)
                exp!(M, q3, q3, X)  # aliased
                @test isapprox(M, q3, q; error = :error, kwargs...)
            end
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
        name = "Logarithmic map on\$M for \$(typeof(p)) points",
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
        name = "Logarithmic map on $M for $(typeof(p)) points",
        kwargs...
    )
    @testset "$(name)" begin
        X = log(M, p, q)
        @test is_vector(M, p, X; error = :error, kwargs...)
        Z = log(M, p, p)
        @test is_vector(M, p, Z; error = :error, kwargs...)
        @test norm(M, p, Z) ≈ 0.0   # log
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
    test_manifold_dimension(
       M; expected_value = nothing, expected_type = Int, name = "Manifold dimension test for \$M",
    )

Test that the dimension of the manifold `M` is consistent.

* that the dimension is nonnegative
* that it is an integer (of type `expected_type`, `Int` by default)
* that it matches the expected value (if provided)
"""
function Manifolds.Test.test_manifold_dimension(
        M;
        expected_value = nothing,
        expected_type = Int,
        name = "Manifold dimension test for $M",
    )
    @testset "$(name)" begin
        d = manifold_dimension(M)
        @test d ≥ 0
        @test isa(d, expected_type)
        isnothing(expected_value) || (@test d == expected_value)
    end
    return nothing
end # Manifolds.Test.test_manifold_dimension

include("tests_general.jl")

end
