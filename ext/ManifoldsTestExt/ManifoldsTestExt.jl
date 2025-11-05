module ManifoldsTestExt

using Manifolds
using ManifoldsBase
using Test

function __init__()
    # Slightly hacked, since we can not extend submodules for now, discuss, what is preferrable
    setglobal!(Manifolds, :Test, Test)
    return nothing
end

module Test
    using Manifolds, Test, ManifoldsBase

    """
        text_exp(
            M, p, X;
            available_functions=[], expected_value=nothing, test_mutating=true, kwargs...
        )
        Test the exponential map on manifold `M` at point `p` with tangent vector `X`.

        * that the result is a valid point on the manifold
        * that the result matches `expected_value`, if given
        * that the mutating version `exp!` matches the non-mutating version, if activated
        * that the logarithmic map inverts the exponential map, if `log` is available
          (only performed if either `injectivity_radius` is not available or `X` is within)
    """
    function test_exp(
            M::AbstractManifold, p, X;
            available_functions = Function[],
            expected_value = nothing,
            test_mutating = true,
            kwargs...
        )
        @testset "Exponential map on $M with point $(typeof(p))" begin
            q = exp(M, p, X)
            @test is_point(M, q; error = :error, kwargs...)
            if expected_value !== nothing
                @test isapprox(M, q, expected_value; error = :error, kwargs...)
            end
            if test_mutating
                q2 = allocate(q)
                exp!(M, q2, p, X)
                @test isapprox(M, q2, q; error = :error, kwargs...)
            end
            if log in available_functions
                # Test only if inj is not available of X is within inj radius
                if !(injectivity_radius in available_functions) || norm(M, p, X) ≤ injectivity_radius(M, p)
                    Y = log(M, p, q)
                    @test isapprox(M, X, Y; error = :error, kwargs...)
                end
            end
        end
        return nothing
    end # Manifolds.Test.test_exp
end
# old includes
include("tests_general.jl")

end
