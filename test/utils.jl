using Manifolds

using LinearAlgebra
using Distributions
using DoubleFloats
using ForwardDiff
using Random
using ReverseDiff
using StaticArrays
using SimpleTraits
using Test

"""
    test_manifold(m::Manifold, pts::AbstractVector;
        test_forward_diff = true,
        retraction_methods = [],
        inverse_retraction_methods = [])

Tests general properties of manifold `m`, given at least three different points
that lie on it (contained in `pts`).

# Arguments
- `test_forward_diff = true`: if true, automatic differentiation using ForwardDiff is
tested.
- `retraction_methods = []`: retraction methods that will be tested.
"""
function test_manifold(M::Manifold, pts::AbstractVector;
    test_forward_diff = true,
    test_reverse_diff = true,
    test_tangent_vector_broadcasting = true,
    test_project_tangent = false,
    retraction_methods = [],
    inverse_retraction_methods = [],
    point_distributions = [],
    tvector_distributions = [],
    exp_log_atol_multiplier = 1)
    # log/exp
    length(pts) ≥ 3 || error("Not enough points (at least three expected)")
    isapprox(M, pts[1], pts[2]) && error("Points 1 and 2 are equal")
    isapprox(M, pts[1], pts[3]) && error("Points 1 and 3 are equal")

    @testset "dimension" begin
        @test isa(manifold_dimension(M), Integer)
        @test manifold_dimension(M) ≥ 0
    end

    @testset "representation" begin
        for T ∈ (Manifolds.MPoint, Manifolds.TVector, Manifolds.CoTVector)
            repr = Manifolds.representation_size(M, T)
            @test isa(repr, Tuple)
            for rs ∈ repr
                @test rs > 0
            end
        end
    end

    @testset "injectivity radius" begin
        @test injectivity_radius(M, pts[1]) > 0
        for rm ∈ retraction_methods
            @test injectivity_radius(M, pts[1], rm) > 0
            @test injectivity_radius(M, pts[1], rm) ≤ injectivity_radius(M, pts[1])
        end
    end

    tv1 = log(M, pts[1], pts[2])

    @testset "is_manifold_point / is_tangent_vector" begin
        for pt ∈ pts
            @test is_manifold_point(M, pt)
        end
        @test is_tangent_vector(M, pts[1], tv1)
    end

    @testset "log/exp tests" begin
        tv2 = log(M, pts[2], pts[1])
        @test isapprox(M, pts[2], exp(M, pts[1], tv1))
        @test isapprox(M, pts[1], exp(M, pts[1], tv1, 0))
        @test isapprox(M, pts[2], exp(M, pts[1], tv1, 1))
        @test isapprox(M, pts[1], exp(M, pts[2], tv2))
        @test is_manifold_point(M, retract(M, pts[1], tv1))
        @test isapprox(M, pts[1], retract(M, pts[1], tv1, 0))
        for retr_method ∈ retraction_methods
            @test is_manifold_point(M, retract(M, pts[1], tv1, retr_method))
            @test isapprox(M, pts[1], retract(M, pts[1], tv1, 0, retr_method))
        end
        new_pt = exp(M, pts[1], tv1)
        retract!(M, new_pt, pts[1], tv1)
        @test is_manifold_point(M, new_pt)
        for x ∈ pts
            @test isapprox(M, zero_tangent_vector(M, x), log(M, x, x); atol = eps(eltype(x)) * exp_log_atol_multiplier)
            @test isapprox(M, zero_tangent_vector(M, x), inverse_retract(M, x, x); atol = eps(eltype(x)) * exp_log_atol_multiplier)
            for inv_retr_method ∈ inverse_retraction_methods
                @test isapprox(M, zero_tangent_vector(M, x), inverse_retract(M, x, x, inv_retr_method); atol = eps(eltype(x)) * exp_log_atol_multiplier)
            end
        end
        zero_tangent_vector!(M, tv1, pts[1])
        @test isapprox(M, pts[1], tv1, zero_tangent_vector(M, pts[1]))
        log!(M, tv1, pts[1], pts[2])
        @test norm(M, pts[1], tv1) ≈ sqrt(inner(M, pts[1], tv1, tv1))

        @test isapprox(M, exp(M, pts[1], tv1, 1), pts[2])
        @test isapprox(M, exp(M, pts[1], tv1, 0), pts[1])

        @test distance(M, pts[1], pts[2]) ≈ norm(M, pts[1], tv1)
    end

    @testset "basic linear algebra in tangent space" begin
        @test isapprox(M, pts[1], 0*tv1, zero_tangent_vector(M, pts[1]))
        @test isapprox(M, pts[1], 2*tv1, tv1+tv1)
        @test isapprox(M, pts[1], 0*tv1, tv1-tv1)
        @test isapprox(M, pts[1], (-1)*tv1, -tv1)
    end

    test_tangent_vector_broadcasting && @testset "broadcasted linear algebra in tangent space" begin
        @test isapprox(M, pts[1], 3*tv1, 2 .* tv1 .+ tv1)
        @test isapprox(M, pts[1], -tv1, tv1 .- 2 .* tv1)
        @test isapprox(M, pts[1], -tv1, .-tv1)
        v = similar(tv1)
        v .= 2 .* tv1 .+ tv1
        @test v ≈ 3*tv1
    end

    test_project_tangent && @testset "project_tangent test" begin
        @test isapprox(M, pts[1], tv1, project_tangent(M, pts[1], tv1))
        tv = similar(tv1)
        project_tangent!(M, tv, pts[1], tv1)
        @test isapprox(M, pts[1], tv, tv1)
    end

    test_forward_diff && @testset "ForwardDiff support" begin
        exp_f(t) = distance(M, pts[1], exp(M, pts[1], t*tv1))
        d12 = distance(M, pts[1], pts[2])
        for t ∈ 0.1:0.1:0.9
            @test d12 ≈ ForwardDiff.derivative(exp_f, t)
        end

        retract_f(t) = distance(M, pts[1], retract(M, pts[1], t*tv1))
        for t ∈ 0.1:0.1:0.9
            @test ForwardDiff.derivative(retract_f, t) ≥ 0
        end
    end

    test_reverse_diff && @testset "ReverseDiff support" begin
        exp_f(t) = distance(M, pts[1], exp(M, pts[1], t[1]*tv1))
        d12 = distance(M, pts[1], pts[2])
        for t ∈ 0.1:0.1:0.9
            @test d12 ≈ ReverseDiff.gradient(exp_f, [t])[1]
        end

        retract_f(t) = distance(M, pts[1], retract(M, pts[1], t[1]*tv1))
        for t ∈ 0.1:0.1:0.9
            @test ReverseDiff.gradient(retract_f, [t])[1] ≥ 0
        end
    end

    @testset "eltype" begin
        tv1 = log(M, pts[1], pts[2])
        @test eltype(tv1) == eltype(pts[1])
        @test eltype(exp(M, pts[1], tv1)) == eltype(pts[1])
    end

    @testset "point distributions" begin
        for pd ∈ point_distributions
            for _ in 1:10
                @test is_manifold_point(M, rand(pd))
            end
        end
    end

    @testset "tangent vector distributions" begin
        for tvd ∈ tvector_distributions
            supp = Manifolds.support(tvd)
            for _ in 1:10
                @test is_tangent_vector(M, supp.x, rand(tvd))
            end
        end
    end
end
