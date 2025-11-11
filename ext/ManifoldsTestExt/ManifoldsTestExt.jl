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
* `:InvalidPoints` is a vector of points that are not on `M`, e.g. to test `is_point`
* `:InvalidVectors` is a vector of tangent vectors that are not in the tangent space of the first point from `:Points`
* `:InverseRetractionMethods` is a vector of inverse retraction methods to test on `M`
  these should have the same order as `:RetractionMethods` (use `nothing` for skipping one)
* `:Points` is a vector of at least 2 points on `M`, which should not be the same point
* `:Mutating` is a boolean (`true` by default) whether to test the mutating variants of functions or not.
* `:Name` is a name of the test. If not provided, defaults to `"\$M"`
* `:RetractionMethods` is a vector of retraction methods to test on `M`
  these should have the same order as `:InverseRetractionMethods` (use `nothing` for skipping one)
* `:Vectors` is a vector of at least 2 tangent vectors, which should be in the tangent space of the correspondinig point entr in `:Points`
* `:VectorTransportMethods` is a vector of vector transport methods to test on `M`
* `:TestWarn` is a boolean (`true` by default) whether to test that whether `error=:warn` in verification functions issues warning.
* `:TestInfo` is a boolean (`true` by default) whether to test that whether `error=:info` in verification functions issues info messages.

Possible entries of the `expectations` dictionary are

* any function tested to provide their expected resulting value, e.g. `exp => p` for the result of `exp(M, p, X)`
* for retractions, inverse retractions, and vector transports, the key is a tuple of the function and the method, e.g. `(retract, method) => q`
* `:atol => 0.0` a global absolute tolerance
* `:atols -> Dict()` a dictionary `function -> atol` for tolerances of specific function tested.
* `:Types` -> Dict() a dictionary `function -> Type` for specifying expected types of results of specific functions, for example `manifold_dimension => Int`.
*  `:IsPointErrors` is a vector of expected error types for each invalid point provided in `:InvalidPoints`, use `nothing` to skip testing for errors for a specific point.
*  `:IsVectorErrors` is a vector of expected error types for each invalid vector provided in `:InvalidVectors`, use `nothing` to skip testing for errors for a specific vector.
*  `:IsVectorBasepointError` is an expected error type when the base point is invalid e.g. for `is_vector`
"""
function Manifolds.Test.test_manifold(M::AbstractManifold, properties::Dict, expectations::Dict = Dict())
    atol = get(expectations, :atol, 0.0)
    mutating = get(properties, :Mutating, true)
    aliased = get(properties, :Aliased, mutating)
    functions = get(properties, :Functions, Function[])
    points = get(properties, :Points, [])
    vectors = get(properties, :Vectors, [])
    test_name = get(properties, :Name, "Manifolds.Test suite for $M")
    test_warn = get(properties, :TestWarn, true)
    test_info = get(properties, :TestInfo, true)
    function_atols = get(expectations, :atols, Dict())
    result_types = get(expectations, :Types, Dict())
    retraction_methods = get(properties, :RetractionMethods, [])
    inverse_retraction_methods = get(properties, :InverseRetractionMethods, [])
    vector_transport_methods = get(properties, :VectorTransportMethods, [])
    t = Test.@testset "$test_name" begin # COV_EXCL_LINE
        n_points = length(points)
        n_vectors = length(vectors)
        if (copy in functions)
            Manifolds.Test.test_copy(
                M, points[1], vectors[1];
                name = "copy(M, p) & copy(M, p, X)",
            )
        end
        if (copyto! in functions)
            Manifolds.Test.test_copyto(
                M, points[1], vectors[1];
                name = "copyto!(M, q, p) & copyto!(M, Y, p, X)",
            )
        end
        if (default_inverse_retraction_method in functions)
            Manifolds.Test.test_default_inverse_retraction(
                M;
                expected_value = get(expectations, default_inverse_retraction_method, nothing),
                name = "default_inverse_retraction_method(M)",
            )
            Manifolds.Test.test_default_inverse_retraction(
                M, typeof(points[1]);
                expected_value = get(expectations, default_inverse_retraction_method, nothing),
                name = "default_inverse_retraction_method(M, P)",
            )
        end
        if (default_retraction_method in functions)
            Manifolds.Test.test_default_retraction(
                M;
                expected_value = get(expectations, default_retraction_method, nothing),
                name = "default_retraction_method(M)",
            )
            Manifolds.Test.test_default_retraction(
                M, typeof(points[1]);
                expected_value = get(expectations, default_retraction_method, nothing),
                name = "default_retraction_method(M, P)",
            )
        end
        if (default_vector_transport_method in functions)
            Manifolds.Test.test_default_vector_transport_method(
                M;
                expected_value = get(expectations, default_vector_transport_method, nothing),
                name = "default_vector_transport_method(M)",
            )
            Manifolds.Test.test_default_vector_transport_method(
                M, typeof(points[1]);
                expected_value = get(expectations, default_vector_transport_method, nothing),
                name = "default_vector_transport_method(M, P)",
            )
        end
        if (distance in functions)
            Manifolds.Test.test_distance(
                M, points[1], points[2];
                available_functions = functions,
                expected_value = get(expectations, distance, nothing),
                name = "distance(M, p, q)", # shorten name within large suite
                atol = get(function_atols, distance, atol),
            )
        end
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
        if (injectivity_radius in functions)
            expected = get(expectations, (injectivity_radius, points[1]), nothing)
            expected_global = get(expectations, injectivity_radius, nothing)
            Manifolds.Test.test_injectivity_radius(
                M, points[1];
                expected_value = expected,
                expected_global_value = expected_global,
                name = "injectivity_radius(M, p)", # shorten name within large suite
            )
            for rm in retraction_methods
                isnothing(rm) && continue
                expected_rm = get(expectations, (injectivity_radius, points[1], rm), nothing)
                expected_rm_global = get(expectations, (injectivity_radius, rm), nothing)
                Manifolds.Test.test_injectivity_radius(
                    M, points[1];
                    expected_value = expected_rm,
                    expected_global_value = expected_rm_global,
                    retraction_method = rm,
                    name = "injectivity_radius(M, p, $rm)", # shorten name within large suite
                )
            end

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
        if (is_point in functions)
            qs = get(properties, :InvalidPoints, [])
            errs = get(expectations, :IsPointErrors, [])
            Manifolds.Test.test_is_point(
                M, points[1], qs...;
                errors = errs,
                name = "is_point on M for $(typeof(points[1])) points",
                test_warn = test_warn,
                test_info = test_info,
                atol = get(function_atols, is_point, atol),
            )
        end
        if (is_vector in functions)
            Ys = get(properties, :InvalidVectors, [])
            errs = get(expectations, :IsVectorErrors, [])
            Manifolds.Test.test_is_vector(
                M, points[1], vectors[1], Ys...;
                basepoint_error = get(expectations, :IsVectorBasepointError, nothing),
                errors = errs,
                name = "is_vector(M, p, X)",
                test_warn = test_warn,
                test_info = test_info,
                q = get(properties, :InvalidPoints, [nothing])[1],
                atol = get(function_atols, is_vector, atol),
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
        if (parallel_transport_to in functions)
            expected_pt = get(expectations, parallel_transport_to, nothing)
            expected_ptd = get(expectations, parallel_transport_direction, nothing)
            Manifolds.Test.test_parallel_transport(
                M, points[1], vectors[1], points[2];
                available_functions = functions,
                expected_value = expected_pt,
                expected_value_direction = expected_ptd,
                test_aliased = aliased,
                test_mutating = (parallel_transport_to! in functions) ? true : mutating,
                atol = get(function_atols, parallel_transport_to, atol),
                name = "parallel_transport_to(M, p, X, q)", # shorten name within large suite
            )
        end
        if (repr in functions)
            expected_repr = get(expectations, repr, nothing)
            Manifolds.Test.test_repr(
                M;
                expected_value = expected_repr,
                name = "repr(M)",
            )
        end
        if (retract in functions)
            for (rm, irm) in zip(retraction_methods, inverse_retraction_methods)
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
        if (vector_transport_to in functions)
            for vtm in vector_transport_methods
                expected_vt = get(expectations, (vector_transport_to, vtm), nothing)
                expected_vtd = get(expectations, (vector_transport_direction, vtm), nothing)
                Manifolds.Test.test_vector_transport(
                    M, points[1], vectors[1], points[2], vtm;
                    available_functions = functions,
                    expected_value = expected_vt,
                    expected_value_direction = expected_vtd,
                    test_aliased = aliased,
                    test_mutating = (vector_transport_to! in functions) ? true : mutating,
                    atol = get(function_atols, vector_transport_to, atol),
                    name = "vector_transport_to(M, p, X, q, $vtm)", # shorten name within large suite
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
    return t
end

# Single function tests
#
# ------------------------------------------------------------------------------------------
"""
    Manifolds.Test.test_copy(M, p, X;
        name = "copying on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test `copy(M, p)` and `copy(M, p, X)` on a manifold `M`.

* that the copied point/vector is a valid point/vector on the manifold / tangent space
* that the copied point/vector matches the original point/vector but is new memory
"""
function Manifolds.Test.test_copy(
        M::AbstractManifold, p, X;
        name = "copying on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        q = copy(M, p)
        Test.@test is_point(M, q; error = :error, kwargs...)
        Test.@test q == p
        Test.@test q !== p
        Y = copy(M, p, X)
        Test.@test is_vector(M, p, Y; error = :error, kwargs...)
        Test.@test Y == X
        Test.@test Y !== X

    end
    return nothing
end # end of Manifolds.Test.test_copy

"""
    Manifolds.Test.test_copyto(M, p, X;
        name = "copying on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test `copyto!(M, q, p)` and `copyto!(M, Y, p, X)` on a manifold `M`.

* that the copied point/vector is a valid point/vector on the manifold / tangent space
* that the copied point/vector matches the original point/vector and is the same memory

Note that since this function does not modify its input, is is called `test_copyto`.
"""
function Manifolds.Test.test_copyto(
        M::AbstractManifold, p, X;
        name = "copyto! on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        # Allocate memory for copyto!
        q = allocate_result(M, exp, p, X)
        q2 = copyto!(M, q, p)
        Test.@test q == p
        Test.@test q2 === q
        Y = allocate_result(M, exp, p, X)
        Y2 = copyto!(M, Y, p, X)
        Test.@test Y == X
        Test.@test Y2 === Y
    end
    return nothing
end

"""
    Manifolds.Test.test_default_inverse_retraction_method(
        M, T=nothing;
        expected_value = nothing,
        expected_type = isnothing(expected_value) ? nothing : typeof(expected_value),
        name = "default_inverse_retraction_method on \$M \$(isnothing(T) ? "" : "for type \$T")",
    )

Test the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method`) on manifold `M`.
* that it returns an [`AbstractInverseRetractionMethod`](@extref `ManifoldsBase.AbstractInverseRetractionMethod`)
* that the result matches `expected_value`, if given
* that the result is of type `expected_type`, if given, defaults to the type of the value
"""
function Manifolds.Test.test_default_inverse_retraction(
        M::AbstractManifold, T = nothing;
        expected_value = nothing,
        expected_type = isnothing(expected_value) ? nothing : typeof(expected_value),
        name = "default_inverse_retraction_method on $M $(isnothing(T) ? "" : "for type $T")",
    )
    Test.@testset "$(name)" begin
        m = isnothing(T) ? default_inverse_retraction_method(M) : default_inverse_retraction_method(M, T)
        Test.@test m isa AbstractInverseRetractionMethod
        isnothing(expected_value) || Test.@test m == expected_value
        isnothing(expected_type) || Test.@test m isa expected_type
    end
    return nothing
end # Manifolds.Test.test_default_inverse_retraction

"""
    Manifolds.Test.test_default_retraction_method(
        M, T=nothing;
        expected_value = nothing,
        expected_type = isnothing(expected_value) ? nothing : typeof(expected_value),
        name = "default_retraction_method on \$M \$(isnothing(T) ? "" : "for type \$T")",
    )

Test the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method`) on manifold `M`.
* that it returns an [`AbstractRetractionMethod`](@extref `ManifoldsBase.AbstractRetractionMethod`)
* that the result matches `expected_value`, if given
* that the result is of type `expected_type`, if given, defaults to the type of the value
"""
function Manifolds.Test.test_default_retraction(
        M::AbstractManifold, T = nothing;
        expected_value = nothing,
        expected_type = isnothing(expected_value) ? nothing : typeof(expected_value),
        name = "default_retraction_method on $M $(isnothing(T) ? "" : "for type $T")",
    )
    Test.@testset "$(name)" begin
        m = isnothing(T) ? default_retraction_method(M) : default_retraction_method(M, T)
        Test.@test m isa AbstractRetractionMethod
        isnothing(expected_value) || Test.@test m == expected_value
        isnothing(expected_type) || Test.@test m isa expected_type
    end
    return nothing
end # Manifolds.Test.test_default_retraction

"""
    Manifolds.Test.test_default_vector_transport_method(
        M, T=nothing;
        expected_value = nothing,
        expected_type = isnothing(expected_value) ? nothing : typeof(expected_value),
        name = "default_vector_transport_method on \$M \$(isnothing(T) ? "" : "for type \$T")",
    )

Test the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method`) on manifold `M`.
* that it returns an [`AbstractVectorTransportMethod`](@extref `ManifoldsBase.AbstractVectorTransportMethod`)
* that the result matches `expected_value`, if given
* that the result is of type `expected_type`, if given, defaults to the type of the value
"""
function Manifolds.Test.test_default_vector_transport_method(
        M::AbstractManifold, T = nothing;
        expected_value = nothing,
        expected_type = isnothing(expected_value) ? nothing : typeof(expected_value),
        name = "default_vector_transport_method on $M $(isnothing(T) ? "" : "for type $T")",
    )
    Test.@testset "$(name)" begin
        m = isnothing(T) ? default_vector_transport_method(M) : default_vector_transport_method(M, T)
        Test.@test m isa AbstractVectorTransportMethod
        isnothing(expected_value) || Test.@test m == expected_value
        isnothing(expected_type) || Test.@test m isa expected_type
    end
    return nothing
end # Manifolds.Test.test_default_vector_transport
"""
    Manifolds.Test.test_distance(M, p, q;
        available_functions=[], expected_value=nothing,
        name = "Distance on \$M between \$(typeof(p)) points",
        kwargs...
    )

Test the distance function on manifold `M` between points `p` and `q`.

* that the result is a nonnegative number
* that the distance from `p` to `p` is zero
* that the distance is symmetric
* that the result matches `expected_value`, if given
* that the distance is equal to the norm of the logarithmic map (if `log` and `norm` are available)
    (only performed if either `injectivity_radius` is not available or the points are within)
"""
function Manifolds.Test.test_distance(
        M::AbstractManifold, p, q;
        available_functions = Function[], expected_value = nothing,
        name = "Distance on $M between $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        d = distance(M, p, q)
        Test.@test d ≥ 0.0
        d_pp = distance(M, p, p)
        Test.@test isapprox(d_pp, 0.0; kwargs...)
        d_qp = distance(M, q, p)
        Test.@test isapprox(d, d_qp; kwargs...)
        if !isnothing(expected_value)
            Test.@test isapprox(d, expected_value; kwargs...)
        end
        if (log in available_functions) && (norm in available_functions)
            # Test only if inj is not available of points are within inj radius
            run_test = !((injectivity_radius in available_functions)) || (d ≤ injectivity_radius(M, p))
            run_test || (@warn("Skipping distance-norm-log test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"))
            Y = log(M, p, q)
            n = norm(M, p, Y)
            Test.@test isapprox(d, n; kwargs...) skip = !run_test
        end
    end
    return nothing
end
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
            run_test || (@warn("Skipping log-exp test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"))
            Y = log(M, p, q)
            Test.@test isapprox(M, X, Y; error = :error, kwargs...) skip = !run_test
        end
    end
    return nothing
end # Manifolds.Test.test_exp

"""
    Manifolds.Test.test_injectority_radius(M, p = nothing;
        expected_value=nothing,
        expected_global_value=nothing,
        retraction_method = nothing,
        name = "Injectivity radius on \$M at point \$(typeof(p)) and \$(retraction_method)",
        kwargs...
    )

Test the injectivity radius on manifold `M` at point `p`.

* that the result is a nonnegative real number
* that the result matches `expected_value`, if given
* if a point `p` is given, that the result is larger or equal to the global injectivity radius
"""
function Manifolds.Test.test_injectivity_radius(
        M::AbstractManifold, p = nothing;
        expected_value = nothing,
        expected_global_value = nothing,
        name = "Injectivity radius on $M at point $(typeof(p))",
        retraction_method = nothing,
        kwargs...
    )
    Test.@testset "$(name)" begin
        r = if isnothing(p)
            isnothing(retraction_method) ? injectivity_radius(M) : injectivity_radius(M, retraction_method)
        else
            isnothing(retraction_method) ? injectivity_radius(M, p) : injectivity_radius(M, p, retraction_method)
        end
        Test.@test r ≥ 0.0
        if !isnothing(expected_value)
            Test.@test isapprox(r, expected_value; kwargs...)
        end
        if !isnothing(p)
            r_global = isnothing(retraction_method) ? injectivity_radius(M) : injectivity_radius(M, retraction_method)
            Test.@test r ≥ r_global
        end
    end
    return nothing
end # Manifolds.Test.test_injectivity_radius

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
    Manifolds.Test.is_point(
        M, p qs...;
        errors = [],
        name = "is_point on \$M for \$(typeof(p)) points",
        test_warn = true,
        test_info = true,
        kwargs...
    )

Test the function [`is_point`](@ref) on  for point `p` on manifold `M`.

* that for `p` it returns `true`.
* that for each `q` in `qs` it
  * returns `false`
  * issues a warning (if activated)
  * isues an info message (if activated)
  * throws the corresponding error from `error_types` (if not nothing)

"""
function Manifolds.Test.test_is_point(
        M::AbstractManifold, p, qs...;
        errors = [],
        name = "is_point on $M for $(typeof(p)) points",
        test_warn = true,
        test_info = true,
        kwargs...
    )
    Test.@testset "$(name)" begin
        Test.@test is_point(M, p; kwargs...)
        for (i, q) in enumerate(qs)
            Test.@test !is_point(M, q; kwargs...)
            (test_warn) && Test.@test_logs (:warn,) is_point(M, q; error = :warn, kwargs...)
            (test_info) && Test.@test_logs (:info,) is_point(M, q; error = :info, kwargs...)
            if length(errors) >= i && !isnothing(errors[i])
                Test.@test_throws (errors[i]) is_point(M, q; error = :error, kwargs...)
            end
        end
    end
    return nothing
end

"""
    Manifolds.Test.test_is_vector(
        M, p, X, Ys...;
        basepoint_error = nothing,
        check_basepoint = true,
        errors = [],
        name = "is_vector on \$M for \$(typeof(p)) points",
        test_warn = true,
        test_info = true,
        q = nothing,
        kwargs...
    )

Test the function [`is_vector`](@ref) on manifold `M` at point `p` for tangent vector `X`.

* that for `X` it returns `true`.
* that for each `Y` in `Ys` it
    * returns `false`
    * issues a warning (if activated)
    * isues an info message (if activated)
    * throws the corresponding error from `error_types` (if not nothing)
* if `check_basepoint` is `true`, then it checks that
    * for `p` this still returns `true`
    * for the base point `q` it
      * returns `false`
      * issues a warning (if activated)
      * isues an info message (if activated)
      * throws the corresponding error from `error_basepoint` (if activated)
"""
function Manifolds.Test.test_is_vector(
        M::AbstractManifold, p, X, Ys...;
        check_basepoint = true,
        errors = [],
        basepoint_error = nothing,
        name = "is_vector on $M for $(typeof(p)) points",
        test_warn = true,
        test_info = true,
        q = nothing,
        kwargs...
    )
    Test.@testset "$(name)" begin
        Test.@test is_vector(M, p, X; kwargs...)
        for (i, Y) in enumerate(Ys)
            Test.@test !is_vector(M, p, Y; kwargs...)
            (test_warn) && Test.@test_logs (:warn,) is_vector(M, p, Y; error = :warn, kwargs...)
            (test_info) && Test.@test_logs (:info,) is_vector(M, p, Y; error = :info, kwargs...)
            if length(errors) >= i && !isnothing(errors[i])
                Test.@test_throws (errors[i]) is_vector(M, p, Y; error = :error, kwargs...)
            end
        end
        if check_basepoint
            Test.@test is_vector(M, p, X, true; kwargs...)
            if !isnothing(q)
                Test.@test !is_vector(M, q, X, true; kwargs...)
                (test_warn) && Test.@test_logs (:warn,) is_vector(M, q, X, true; error = :warn, kwargs...)
                (test_info) && Test.@test_logs (:info,) is_vector(M, q, X, true; error = :info, kwargs...)
                if !isnothing(basepoint_error)
                    @info basepoint_error
                    Test.@test_throws (basepoint_error) is_vector(M, q, X, true; error = :error, kwargs...)
                end
            end
        end
    end
    return nothing
end

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
            run_test || (@warn("Skipping exp-log test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"))
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
    Manifolds.Test.test_parallel_transport(
        M, p, X, q;
        available_functions=[],
        direction = (log in available_functions) ? log(M, p, q) : nothing,
        expected_value=nothing,
        expected_value_direction=nothing,
        name = "Parallel transport on \$M at point \$(typeof(p))",
        test_aliased = true,
        test_mutating = true,
        kwargs...
    )

Test the parallel transport on manifold `M` at point `p` with tangent vector `X` to a point `q`.

this defaults to testing `parallel_transport_to`, but can also be used to test `parallel_transport_direction`,
by passing that function to the `available_functions`, using the `direction` argument.
The `direction` has to be the one that transports `X` also to `q`.

* that the result is a valid tangent vector at `q` on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `parallel_transport!` matches the non-mutating version
* that `parallel_transport_to` works on aliased in put (`Y=X`) (if activated)
* that the direction of the transport is consistent with `expected_direction`, if given
* that the mutating version `parallel_transport_direction!` matches the non-mutating version (if activated)
* that `parallel_transport_direction` works on aliased in put (`Y=X` or `Y=d`) (if activated for mutating)
* that both functions are consistent
"""
function Manifolds.Test.test_parallel_transport(
        M, p, X, q;
        available_functions = [],
        direction = (log in available_functions) ? log(M, p, q) : nothing,
        expected_value_direction = nothing,
        expected_value = nothing,
        name = "Parallel transport on $M at point $(typeof(p))",
        test_aliased = true,
        test_mutating = true,
        kwargs...
    )
    Test.@testset "$(name)" begin
        Y = parallel_transport_to(M, p, X, q)
        Test.@test is_vector(M, q, Y; error = :error, kwargs...)
        if !isnothing(expected_value)
            Test.@test isapprox(M, p, Y, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            Y2 = copy(M, p, X)
            parallel_transport_to!(M, Y2, p, X, q)
            Test.@test isapprox(M, q, Y2, Y; error = :error, kwargs...)
            if test_aliased
                Y3 = copy(M, p, X)
                parallel_transport_to!(M, Y3, p, Y3, q)  # aliased
                Test.@test isapprox(M, q, Y3, Y; error = :error, kwargs...)
            end
        end
        if (parallel_transport_direction in available_functions) && !isnothing(direction)
            Y4 = parallel_transport_direction(M, p, X, direction)
            if !isnothing(expected_value_direction)
                Test.@test isapprox(M, q, Y4, expected_value_direction; error = :error, kwargs...)
            end
            if test_mutating
                Y5 = copy(M, p, X)
                parallel_transport_direction!(M, Y5, p, X, direction)
                Test.@test isapprox(M, p, Y5, Y4; error = :error, kwargs...)
                if test_aliased
                    Y6 = copy(M, p, X)
                    parallel_transport_direction!(M, Y6, p, Y6, direction)  # aliased #1
                    Test.@test isapprox(M, p, Y6, Y4; error = :error, kwargs...)
                    Y7 = copy(M, p, direction)
                    parallel_transport_direction!(M, Y7, p, X, direction)  # aliased #2
                    Test.@test isapprox(M, p, Y7, Y4; error = :error, kwargs...)
                end
            end
            # consistency check
            Test.@test is_vector(M, q, Y4; error = :error, kwargs...)
            Test.@test isapprox(M, q, Y, Y4; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_parallel_transport

"""
    Manifolds.Test.test_repr(
        M;
        expected_value=nothing,
        name = "(String) repr_esentation of \$M",
    )

Test that the default `show` method works as expected by calling `repr(M)`.
"""
function Manifolds.Test.test_repr(
        M; expected_value = nothing, name = "(String) repr_esentation of \$M",
    )
    Test.@testset "$(name)" begin
        s = repr(M)
        if !isnothing(expected_value)
            Test.@test s == expected_value
        end
    end
    return nothing
end # Manifolds.Test.test_repr

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
    Manifolds.Test.test_vector_transport(
        M, p, X, q, default_vector_transport_method(M);
        available_functions=[],
        direction = inverse_retract(M, p, q, default_inverse_retraction_method(M)),
        expected_value=nothing,
        expected_value_direction=nothing,
        test_aliased = true,
        test_mutating = true,
        name = "Vector transport method \$(vector_transport_method) on \$M at point \$(typeof(p))",
        kwargs...
    )

Test the vector transport on manifold `M` at point `p` with tangent vector `X` to a point `q`.

this defaults to testing `vector_transport_to`, but can also be used to test `vector_transport_direction`,
by passing that function to the `available_functions`, using the `direction` argument.
The `direction` has to be the one that transports `X` also to `q`.

* that the result is a valid tangent vector at `q` on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `vector_transport!` matches the non-mutating version
* that `vector_transport_to` works on aliased in put (`Y=X`) (if activated)
* that the direction of the transport is consistent with `expected_direction`, if given
* that the mutating version `vector_transport_direction!` matches the non-mutating version (if activated)
* that `vector_transport_direction` works on aliased in put (`Y=X` or `Y=d`) (if activated for mutating)
* that both functions are consistent
"""
function Manifolds.Test.test_vector_transport(
        M, p, X, q, m = default_vector_transport_method(M);
        available_functions = [],
        direction = inverse_retract(M, p, q, default_inverse_retraction_method(M)),
        expected_value_direction = nothing,
        expected_value = nothing,
        test_aliased = true,
        test_mutating = true,
        name = "Vector transport method $(vector_transport_method) on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        Y = vector_transport_to(M, p, X, q, m)
        Test.@test is_vector(M, q, Y; error = :error, kwargs...)
        if !isnothing(expected_value)
            Test.@test isapprox(M, p, Y, expected_value; error = :error, kwargs...)
        end
        if test_mutating
            Y2 = copy(M, p, X)
            vector_transport_to!(M, Y2, p, X, q, m)
            Test.@test isapprox(M, q, Y2, Y; error = :error, kwargs...)
            if test_aliased
                Y3 = copy(M, p, X)
                vector_transport_to!(M, Y3, p, Y3, q)  # aliased
                Test.@test isapprox(M, p, Y3, Y; error = :error, kwargs...)
            end
        end
        if (vector_transport_direction in available_functions) && !isnothing(direction)
            Y4 = vector_transport_direction(M, p, X, direction, m)
            if !isnothing(expected_value_direction)
                Test.@test isapprox(M, p, Y4, expected_value_direction; error = :error, kwargs...)
            end
            if test_mutating
                Y5 = copy(M, p, X)
                vector_transport_direction!(M, Y5, p, X, direction, m)
                Test.@test isapprox(M, p, Y5, Y4; error = :error, kwargs...)
                if test_aliased
                    Y6 = copy(M, p, X)
                    vector_transport_direction!(M, Y6, p, Y6, direction, m)  # aliased #1
                    Test.@test isapprox(M, q, Y6, Y4; error = :error, kwargs...)
                    Y7 = copy(M, p, direction)
                    vector_transport_direction!(M, Y7, p, X, direction, m)  # aliased #2
                    Test.@test isapprox(M, q, Y7, Y4; error = :error, kwargs...)
                end
            end
            # consistency check
            Test.@test is_vector(M, q, Y4; error = :error, kwargs...)
            Test.@test isapprox(M, q, Y, Y4; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_vector_transport

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
