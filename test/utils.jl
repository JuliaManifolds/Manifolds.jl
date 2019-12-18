using Manifolds
using ManifoldsBase

using LinearAlgebra
using Distributions
using DoubleFloats
using ForwardDiff
using Random
using ReverseDiff
using StaticArrays
using Test

"""
    test_manifold(m::Manifold, pts::AbstractVector;
        test_forward_diff = true,
        retraction_methods = [],
        inverse_retraction_methods = [])

Tests general properties of manifold `m`, given at least three different points
that lie on it (contained in `pts`).

# Arguments
- `exp_log_atol_multiplier = 0`, change absolute tolerance of exp/log tests (0 use default, i.e. deactivate atol and use rtol)
- `exp_log_rtol_multiplier = 1`, change the relative tolerance of exp/log tests (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.
- `inverse_retraction_methods = []`: inverse retraction methods that will be tested.
- `point_distributions = []` : point distributions to test
- `projection_tvector_atol_multiplier = 0` : chage absolute tolerance in testing projections (0 use default, i.e. deactivate atol and use rtol)
-  tvector_distributions = []` : tangent vector distributions to test
- `rand_tvector_atol_multiplier = 0` : chage absolute tolerance in testing random vectors (0 use default, i.e. deactivate atol and use rtol)
  random tangent vectors are tangent vectors
- `retraction_methods = []`: retraction methods that will be tested.
- `test_forward_diff = true`: if true, automatic differentiation using
  ForwardDiff is tested.
- `test_forward_diff = true`: if true, automatic differentiation using
  ReverseDiff is tested.
- `test_musical_isomorphisms = false` : test musical isomorphisms 
- `test_project_tangent = false` : test projections on tangent spaces
- `test_representation_size = true` : test repersentation size of points/tvectprs
- `test_tangent_vector_broadcasting = true` : test boradcasting operators on TangentSpace
- `test_vector_transport = false` : test vector transport
- `test_mutating_rand = false` : test the mutating random function for points on manifolds
"""
function test_manifold(M::Manifold, pts::AbstractVector;
    test_exp_log = true,
    test_log_yields_tangent = true,
    test_injectivity_radius=true,
    test_forward_diff = true,
    test_reverse_diff = true,
    test_tangent_vector_broadcasting = true,
    test_project_tangent = false,
    test_representation_size = true,
    test_musical_isomorphisms = false,
    test_vector_transport = false,
    test_mutating_rand = false,
    default_inverse_retraction_method = ManifoldsBase.LogarithmicInverseRetraction(),
    retraction_methods = [],
    inverse_retraction_methods = [],
    point_distributions = [],
    tvector_distributions = [],
    exp_log_atol_multiplier = 0,
    exp_log_rtol_multiplier = 1,
    projection_atol_multiplier = 0,
    rand_tvector_atol_multiplier = 0)

    length(pts) ≥ 3 || error("Not enough points (at least three expected)")
    isapprox(M, pts[1], pts[2]) && error("Points 1 and 2 are equal")
    isapprox(M, pts[1], pts[3]) && error("Points 1 and 3 are equal")

    @testset "dimension" begin
        @test isa(manifold_dimension(M), Integer)
        @test manifold_dimension(M) ≥ 0
        @test manifold_dimension(M) == vector_space_dimension(Manifolds.VectorBundleFibers(Manifolds.TangentSpace, M))
        @test manifold_dimension(M) == vector_space_dimension(Manifolds.VectorBundleFibers(Manifolds.CotangentSpace, M))
    end

    test_representation_size && @testset "representation" begin
        function test_repr(repr)
            @test isa(repr, Tuple)
            for rs ∈ repr
                @test rs > 0
            end
        end

        test_repr(Manifolds.representation_size(M))
        for VS ∈ (Manifolds.TangentSpace, Manifolds.CotangentSpace)
            test_repr(Manifolds.representation_size(Manifolds.VectorBundleFibers(VS, M)))
        end
    end

    test_injectivity_radius && @testset "injectivity radius" begin
        @test injectivity_radius(M, pts[1]) > 0
        @test injectivity_radius(M, pts[1]) ≥ injectivity_radius(M)
        for rm ∈ retraction_methods
            @test injectivity_radius(M, pts[1], rm) > 0
            @test injectivity_radius(M, pts[1], rm) ≤ injectivity_radius(M, pts[1])
        end
    end

    @testset "is_manifold_point" begin
        for pt ∈ pts
            @test is_manifold_point(M, pt)
            @test check_manifold_point(M, pt) === nothing
            @test check_manifold_point(M, pt) === nothing
        end
    end

    tv1 = inverse_retract(M,pts[1],pts[2],default_inverse_retraction_method)
    test_log_yields_tangent && @testset "is_tangent_vector" begin
        @test is_tangent_vector(M, pts[1], tv1; atol = eps(eltype(pts[1])))
        @test check_tangent_vector(M, pts[1], tv1; atol = eps(eltype(pts[1]))) === nothing
        @test check_tangent_vector(M, pts[1], tv1; atol = eps(eltype(pts[1]))) === nothing
    end

    test_exp_log && @testset "log/exp tests" begin
        tv1 = log(M, pts[1], pts[2])
        tv2 = log(M, pts[2], pts[1])
        @test isapprox(M, pts[2], exp(M, pts[1], tv1))
        @test isapprox(M, pts[1], exp(M, pts[1], tv1, 0))
        @test isapprox(M, pts[2], exp(M, pts[1], tv1, 1))
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
            @test isapprox(M, x, zero_tangent_vector(M, x), log(M, x, x);
                atol = eps(eltype(x)) * exp_log_atol_multiplier,
                rtol = exp_log_atol_multiplier == 0. ? sqrt(eps(eltype(x)))*exp_log_rtol_multiplier : 0
            )
            @test isapprox(M, x, zero_tangent_vector(M, x), inverse_retract(M, x, x);
                atol = eps(eltype(x)) * exp_log_atol_multiplier,
                rtol = exp_log_atol_multiplier == 0. ? sqrt(eps(eltype(x)))*exp_log_rtol_multiplier : 0.
            )
            for inv_retr_method ∈ inverse_retraction_methods
                @test isapprox(M, x, zero_tangent_vector(M, x), inverse_retract(M, x, x, inv_retr_method);
                    atol = eps(eltype(x)) * exp_log_atol_multiplier,
                    rtol = exp_log_atol_multiplier == 0 ? sqrt(eps(eltype(x)))*exp_log_rtol_multiplier : 0
                )
            end
        end
        zero_tangent_vector!(M, tv1, pts[1])
        @test isapprox(M, pts[1], tv1, zero_tangent_vector(M, pts[1]); atol = eps(eltype(pts[1])) * exp_log_atol_multiplier)
        log!(M, tv1, pts[1], pts[2])
        @test norm(M, pts[1], tv1) ≈ sqrt(inner(M, pts[1], tv1, tv1))

        @test isapprox(M, exp(M, pts[1], tv1, 1), pts[2]; atol = eps(eltype(pts[1])) * exp_log_atol_multiplier)
        @test isapprox(M, exp(M, pts[1], tv1, 0), pts[1]; atol = eps(eltype(pts[1])) * exp_log_atol_multiplier)

        @test distance(M, pts[1], pts[2]) ≈ norm(M, pts[1], tv1)
    end

    @testset "vector spaces tests" begin
        tv = zero_tangent_vector(M, pts[1])
        mts = Manifolds.VectorBundleFibers(Manifolds.TangentSpace, M)
        @test isapprox(M, pts[1], tv, zero_vector(mts, pts[1]))
        zero_vector!(mts, tv, pts[1])
        @test isapprox(M, pts[1], tv, zero_tangent_vector(M, pts[1]))
    end

    @testset "basic linear algebra in tangent space" begin
        @test isapprox(M, pts[1], 0*tv1, zero_tangent_vector(M, pts[1]); atol = eps(eltype(pts[1])))
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
        @test isapprox(M, pts[1], tv1, project_tangent(M, pts[1], tv1); atol = eps(eltype(pts[1])) * projection_atol_multiplier)
        tv = similar(tv1)
        project_tangent!(M, tv, pts[1], tv1)
        @test isapprox(M, pts[1], tv, tv1; atol = eps(eltype(pts[1])) * projection_atol_multiplier)
    end

    test_vector_transport && @testset "vector transport" begin
        v1 = inverse_retract(M, pts[1], pts[2],default_inverse_retraction_method)
        v2 = inverse_retract(M, pts[1], pts[3],default_inverse_retraction_method)
        v1t1 = vector_transport_to(M, pts[1], v1, pts[3])
        v1t2 = vector_transport_direction(M, pts[1], v1, v2)
        @test is_tangent_vector(M, pts[3], v1t1)
        @test is_tangent_vector(M, pts[3], v1t2)
        @test isapprox(M, pts[3], v1t1, v1t2)
        @test isapprox(M, pts[1], vector_transport_to(M, pts[1], v1, pts[1]), v1)
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

    test_musical_isomorphisms && @testset "Musical isomorphisms" begin
        tv_m = inverse_retract(M, pts[1], pts[2],default_inverse_retraction_method)
        ctv_m = flat(M, pts[1], FVector(TangentSpace, tv_m))
        @test ctv_m.type == CotangentSpace
        tv_m_back = sharp(M, pts[1], ctv_m)
        @test tv_m_back.type == TangentSpace
    end

    @testset "eltype" begin
        tv1 = inverse_retract(M, pts[1], pts[2],default_inverse_retraction_method)
        @test eltype(tv1) == eltype(pts[1])
        @test eltype(exp(M, pts[1], tv1)) == eltype(pts[1])
    end

    @testset "copyto!" begin
        p2 = similar(pts[1])
        copyto!(p2, pts[2])
        @test isapprox(M, p2, pts[2])

        tv2 = similar(tv1)
        copyto!(tv2, inverse_retract(M, pts[2], pts[3],default_inverse_retraction_method))
        @test isapprox(M, pts[2], tv2, inverse_retract(M, pts[2], pts[3],default_inverse_retraction_method))
    end

    @testset "point distributions" begin
        prand = similar(pts[1])
        for pd ∈ point_distributions
            for _ in 1:10
                @test is_manifold_point(M, rand(pd))
                if test_mutating_rand
                    rand!(pd, prand)
                    @test is_manifold_point(M, prand)
                end
            end
        end
    end

    @testset "tangent vector distributions" begin
        for tvd ∈ tvector_distributions
            supp = Manifolds.support(tvd)
            for _ in 1:10
                randtv = rand(tvd)
                @test is_tangent_vector(M, supp.x, randtv; atol = rand_tvector_atol_multiplier * eps(eltype(randtv)))
            end
        end
    end
end
