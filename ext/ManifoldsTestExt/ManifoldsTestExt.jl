module ManifoldsTestExt

using Manifolds
using ManifoldsBase
using Test

#
#
#
"""
    test_manifold(G::AbstractManifold, properties::Dict, expectations::Dict)

Test the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) ``\\mathcal M``
based on a `Dict` of properties and a `Dict` of `expectations`.

Three functions are expected to be defined (without explicitly being passed in the `properties`):
`is_point(M, p)`, `is_vector(M, p, X)`, and `isapprox(M, p, q)` / `isapprox(M, p, X, Y)`,
since these are essential for verifying results.

Possible properties are

* `:Aliased` is a boolean (same as `:Mutating` by default) whether to test the mutating variants with aliased input
* `:Functions` is a vector of all defined functions for `M`
  note a test is activated by the function (like `exp`), adding the mutating function (like `exp!`) overwrites the
  global default (see `:Mutating`) to true.
* `:InverseRetractionMethods` is a vector of inverse retraction methods to test on `M`
  these should have the same order as `:RetractionMethods` (use `nothing` for skipping one)
* `:Points` is a vector of at least 2 points on `M`, which should not be the same point
* `:Vectors` is a vector of at least 2 tangent vectors, which should be in the tangent space of the correspondinig point entr in `:Points`
* `:Mutating` is a boolean (`true` by default) whether to test the mutating variants of functions or not.
* `:Name` is a name of the test. If not provided, defaults to `"\$M"`
* `:Vectors` is a vector of at least 2 tangent vectors, which should be in the tangent space of the correspondinig point entr in `:Points`
* `:RetractionMethods` is a vector of retraction methods to test on `M`
  these should have the same order as `:InverseRetractionMethods` (use `nothing` for skipping one)
* `:VectorTransportMethods` is a vector of vector transport methods to test on `M`

Possible entries of the `expectations` dictionary are

* any function tested to provide their expected resulting value, e.g. `exp => p` for the result of `exp(M, p, X)`
* for retractions, inverse retractions, and vector transports, the key is a tuple of the function and the method, e.g. `(retract, method) => q`
* `:atol => 0.0` a global absolute tolerance
* `:atols -> Dict()` a dictionary `function -> atol` for tolerances of specific function tested.
* `:Types` -> Dict() a dictionary `function -> Type` for specifying expected types of results of specific functions, for example `manifold_dimension => Int`.

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
    retraction_methods = get(properties, :RetractionMethods, [])
    inverse_retraction_methods = get(properties, :InverseRetractionMethods, [])
    vector_transport_methods = get(properties, :VectorTransportMethods, [])
    return Test.@testset "$test_name" begin
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
        if (inner in functions)
            expected_inner = get(expectations, inner, nothing)
            Manifolds.Test.test_inner(
                M, points[1], vectors[1], vectors[2];
                available_functions = functions,
                expected_value = expected_inner,
                name = "inner(M, p, X, Y)", # shorten name within large suite
                atol = get(function_atols, inner, atol),
            )
        end
        if (inverse_retract in functions)
            for (irm, rm) in zip(inverse_retraction_methods, retraction_methods)
                isnothing(irm) && continue
                expected_inv_retract = get(expectations, (inverse_retract, irm), nothing)
                Manifolds.Test.test_inverse_retract(
                    M, points[1], points[2], irm;
                    available_functions = functions,
                    expected_value = expected_inv_retract,
                    retraction_method = rm,
                    test_mutating = (inverse_retract! in functions) ? true : mutating,
                    atol = get(function_atols, inverse_retract, atol),
                    name = "inverse_retract(M, p, q, $irm)", # shorten name within large suite
                )
            end
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
        if (norm in functions)
            expected_norm = get(expectations, norm, nothing)
            Manifolds.Test.test_norm(
                M, points[1], vectors[1];
                available_functions = functions,
                expected_value = expected_norm,
                name = "norm(M, p, X)", # shorten name within large suite
                atol = get(function_atols, norm, atol),
            )
        end
        if (retract in functions)
            @info "Testing retract with methods: $retraction_methods"
            for (rm, irm) in zip(retraction_methods, inverse_retraction_methods)
                @warn rm, irm
                isnothing(rm) && continue
                expected_retract = get(expectations, (retract, rm), nothing)
                Manifolds.Test.test_retract(
                    M, points[1], vectors[1], rm;
                    available_functions = functions,
                    expected_value = expected_retract,
                    inverse_retraction_method = irm,
                    test_aliased = aliased,
                    test_mutating = (retract! in functions) ? true : mutating,
                    atol = get(function_atols, retract, atol),
                    name = "retract(M, p, X, $rm)", # shorten name within large suite
                )
            end
        end
        if (zero_vector in functions)
            Manifolds.Test.test_zero_vector(
                M, points[1];
                available_functions = functions,
                test_mutating = (zero_vector! in functions) ? true : mutating,
                atol = get(function_atols, zero_vector, atol),
                name = "zero_vector(M, p)", # shorten name within large suite
            )
        end
    end # end of test_manifold testset
end

# Single function tests
#
# ------------------------------------------------------------------------------------------
"""
    Manifolds.Test.test_exp(
        M, p, X, t=1.0;
        available_functions=[], expected_value=nothing, test_mutating=true,
        test_log = (log in available_functions),
        test_fused = true,
        test_injectivity_radius = (injectivity_radius in available_functions),
        name = "Exponential map on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test the exponential map on manifold `M` at point `p` with tangent vector `X`.

* that the result is a valid point on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `exp!` matches the non-mutating version, (if activated)
* that `exp!` works on aliased in put (`p=q`) (if activated for mutating)
* that the logarithmic map inverts the exponential map (if activated)
(only performed if either `injectivity_radius` is not available or `X` is within)
* that the fused version `exp_fused(M, p, t, X)` matches the non-fused version (if activated)
"""
function Manifolds.Test.test_exp(
        M::AbstractManifold, p, X, t = 1.0;
        available_functions = Function[],
        expected_value = nothing,
        test_aliased = true,
        test_fused = true,
        test_log = (log in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        test_mutating = true,
        name = "Exponential map on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        q = exp(M, p, X)
        Test.@test is_point(M, q; error = :error, kwargs...)
        if !isnothing(expected_value)
            Test.@test isapprox(M, q, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            q2 = copy(M, p)
            exp!(M, q2, p, X)
            Test.@test isapprox(M, q2, q; error = :error, kwargs...)
            if test_aliased
                q3 = copy(M, p)
                exp!(M, q3, q3, X)  # aliased
                Test.@test isapprox(M, q3, q; error = :error, kwargs...)
            end
        end
        if test_fused
            q4 = Manifolds.exp_fused(M, p, X, t)
            q5 = exp(M, p, t * X)
            Test.@test isapprox(M, q4, q5; error = :error, kwargs...)
            if test_mutating
                q6 = copy(M, p)
                Manifolds.exp_fused!(M, q6, p, X, t)
                Test.@test isapprox(M, q6, q5; error = :error, kwargs...)
                if test_aliased
                    q7 = copy(M, p)
                    Manifolds.exp_fused!(M, q7, q7, X, t)  # aliased
                    Test.@test isapprox(M, q7, q5; error = :error, kwargs...)
                end
            end
        end
        if test_log
            # Test only if inj is not available of X is within inj radius
            run_test = !test_injectivity_radius || norm(M, p, X) ≤ injectivity_radius(M, p)
            run_test || (@warn("Skipping log-exp test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"); nothing)
            Y = log(M, p, q)
            Test.@test isapprox(M, X, Y; error = :error, kwargs...) skip = !run_test
        end
    end
    return nothing
end # Manifolds.Test.test_exp
"""
    Manifolds.Test.test_inner(M, p, X, Y;
        available_functions=[], expected_value=nothing,
        name = "Inner product on \$M at point \$(typeof(p))",
        test_norm = (norm in available_functions),
        kwargs...
    )

Test the inner product on the manifold `M` at point `p` for tangent vectors `X` and `Y`.

* that the result is a real number
* that the result matches `expected_value`, if given
* that the inner product of `X` with itself is non negative
* that the inner product of `X` with itself is the same as its norm squared (if activated)
"""
function Manifolds.Test.test_inner(
        M::AbstractManifold, p, X, Y;
        available_functions = Function[],
        expected_value = nothing,
        test_norm = (norm in available_functions),
        name = "Inner product on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        v = inner(M, p, X, Y)
        Test.@test v isa (Real)
        isnothing(expected_value) || Test.@test isapprox(v, expected_value; kwargs...)
        w = inner(M, p, X, X)
        Test.@test w ≥ 0.0
        if test_norm
            n = norm(M, p, X)
            Test.@test isapprox(w, n^2; kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_inner

"""
    Manifolds.Test.test_inverse_retraction(
        M, p, q, m::AbstractInverseRetractionMethod;
        available_functions=[],
        expected_value=nothing,
        name = "Inverse retraction \$m on \$M at point \$(typeof(p))",
        retraction_method=nothing,
        test_mutating = true,
        test_retraction = (retract in available_functions) && !isnothing(retraction_method),
        kwargs...
    )

Test the inverse retraction method `m` on manifold `M` at point `p` towards point `q`.

* that the result is a valid tangent vector at `p` on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `inverse_retract!` matches the non-mutating version, (if activated)
* that the retraction inverts the inverse retraction (if activated)
"""
function Manifolds.Test.test_inverse_retract(
        M, p, q, m::AbstractInverseRetractionMethod;
        available_functions = Function[],
        expected_value = nothing,
        name = "Inverse retraction $m on $M at point $(typeof(p))",
        retraction_method = nothing,
        test_retraction = (retract in available_functions) && !isnothing(retraction_method),
        test_mutating = true,
        kwargs...
    )
    Test.@testset "$(name)" begin
        X = inverse_retract(M, p, q, m)
        Test.@test is_vector(M, p, X; error = :error, kwargs...)
        if !isnothing(expected_value)
            Test.@test isapprox(M, p, X, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            Y = copy(M, p, X)
            inverse_retract!(M, Y, p, q, m)
            Test.@test isapprox(M, p, Y, X; error = :error, kwargs...)
        end
        if test_retraction
            q2 = retract(M, p, X, retraction_method)
            Test.@test isapprox(M, q2, q; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_inverse_retraction

"""
    Manifolds.Test.test_log(
        M, p, q;
        available_functions=[], expected_value=nothing, test_mutating=true,
        test_exp = (exp in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        name = "Logarithmic map on \$M for \$(typeof(p)) points",
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
    Test.@testset "$(name)" begin
        X = log(M, p, q)
        Test.@test is_vector(M, p, X; error = :error, kwargs...)
        Z = log(M, p, p)
        Test.@test is_vector(M, p, Z; error = :error, kwargs...)
        Test.@test norm(M, p, Z) ≈ 0.0   # log
        if !isnothing(expected_value)
            Test.@test isapprox(M, p, X, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            Y = copy(M, p, X)
            log!(M, Y, p, q)
            Test.@test isapprox(M, Y, X; error = :error, kwargs...)
        end
        if test_exp
            # Test only if inj is not available of X is within inj radius
            run_test = !test_injectivity_radius || norm(M, p, X) ≤ injectivity_radius(M, p)
            run_test || (@warn("Skipping exp-log test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"); nothing)
            q2 = exp(M, p, X)
            Test.@test isapprox(M, q2, q; error = :error, kwargs...) skip = !run_test
        end
    end
    return nothing
end # Manifolds.Test.test_log

"""
    test_manifold_dimension(
        M;
        expected_value = nothing,
        expected_type = Int,
        name = "Manifold dimension for \$M",
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
        name = "Manifold dimension for $M",
    )
    Test.@testset "$(name)" begin
        d = manifold_dimension(M)
        Test.@test d ≥ 0
        Test.@test isa(d, expected_type)
        isnothing(expected_value) || (@test d == expected_value)
    end
    return nothing
end # Manifolds.Test.test_manifold_dimension

"""
    Manifolds.Test.test_norm(M, p, X;
        available_functions=[], expected_value=nothing,
        name = "Norm on \$M at point \$(typeof(p))",
        test_inner = (inner in available_functions),
        kwargs...
    )

Test the norm on the manifold `M` at point `p` for tangent vector `X`.

* that the result is a real number
* that the result matches `expected_value`, if given
* that the norm of `X` with itself is non negative
* that the inner product of `X` with itself is the same as its norm squared (if activated)
"""
function Manifolds.Test.test_norm(
        M::AbstractManifold, p, X;
        available_functions = Function[],
        expected_value = nothing,
        test_inner = (inner in available_functions),
        name = "Norm on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        v = norm(M, p, X)
        Test.@test v isa (Real)
        isnothing(expected_value) || Test.@test isapprox(v, expected_value; kwargs...)
        Test.@test v ≥ 0.0
        if test_inner
            w = inner(M, p, X, X)
            Test.@test isapprox(w, v^2; kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_norm

"""
    Manifolds.test.test_retract(
        M, p, X, m::AbstractRetractionMethod;
        available_functions=[],
        expected_value=nothing,
        inverse_retraction_method=nothing,
        name = "Retraction \$m on \$M at point \$(typeof(p))",
        t = 1.0
        test_mutating = true,
        test_mutating = true,
        test_inverse_retraction = (inverse_retraction in available_functions) && !isnothing(inverse_retraction_method),
        kwargs...
    )

Test the retraction method `m` on manifold `M` at point `p` with tangent vector `X`.

* that the result is a valid point on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `retract!` matches the non-mutating version, (if activated)
* that `retract!` works on aliased in put (`p=q`) (if activated for mutating)
* that the inverse retraction inverts the retraction (if activated)
* that the fused version `retract_fused(M, p, X, t, m)` matches the non-fused version (if activated)
"""
function Manifolds.Test.test_retract(
        M, p, X, m::AbstractRetractionMethod;
        t = 1.0,
        available_functions = Function[],
        expected_value = nothing,
        inverse_retraction_method = nothing,
        name = "Retraction $m on $M at point $(typeof(p))",
        test_aliased = true,
        test_inverse_retraction = (inverse_retract in available_functions) && !isnothing(inverse_retraction_method),
        test_mutating = true,
        test_fused = true,
        kwargs...
    )
    Test.@testset "$(name)" begin
        q = retract(M, p, X, m)
        Test.@test is_point(M, q; error = :error, kwargs...)
        if !isnothing(expected_value)
            Test.@test isapprox(M, q, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            q2 = copy(M, p)
            retract!(M, q2, p, X, m)
            Test.@test isapprox(M, q2, q; error = :error, kwargs...)
            if test_aliased
                q3 = copy(M, p)
                retract!(M, q3, q3, X, m)  # aliased
                Test.@test isapprox(M, q3, q; error = :error, kwargs...)
            end
        end
        if test_fused
            q4 = Manifolds.retract_fused(M, p, X, t, m)
            q5 = retract(M, p, t * X, m)
            Test.@test isapprox(M, q4, q5; error = :error, kwargs...)
            if test_mutating
                q6 = copy(M, p)
                Manifolds.retract_fused!(M, q6, p, X, t, m)
                Test.@test isapprox(M, q6, q5; error = :error, kwargs...)
                if test_aliased
                    q7 = copy(M, p)
                    Manifolds.retract_fused!(M, q7, q7, X, t, m)  # aliased
                    Test.@test isapprox(M, q7, q5; error = :error, kwargs...)
                end
            end
        end
        if test_inverse_retraction
            X2 = inverse_retract(M, p, q, inverse_retraction_method)
            Test.@test isapprox(M, p, X2, X; error = :error, kwargs...)

        end
    end
    return nothing
end # Manifolds.Test.test_retract

"""
    Manifolds.Test.test_zero_vector(M, p;
        available_functions=[],
        test_mutating = true,
        test_norm = (norm in available_functions),
        name = "Zero vector test on \$M at point \$(typeof(p))",
    )

Test the zero vector on the manifold `M` at point `p`.
* verify that it is a valid tangent vector
* verify that its norm is zero (if provided)
* verify that the mutating version `zero_vector!` matches the non-mutating version (if activated)
"""
function Manifolds.Test.test_zero_vector(
        M::AbstractManifold, p;
        available_functions = Function[],
        test_mutating = true,
        test_norm = (norm in available_functions),
        name = "Zero vector on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        Z = zero_vector(M, p)
        Test.@test is_vector(M, p, Z; error = :error, kwargs...)
        if test_norm
            n = norm(M, p, Z)
            Test.@test isapprox(n, 0.0; kwargs...)
        end
        if test_mutating
            Z2 = copy(M, p)
            zero_vector!(M, Z2, p)
            Test.@test isapprox(M, Z2, Z; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_zero_vector


include("tests_general.jl")

end
