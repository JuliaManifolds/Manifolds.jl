include("../utils.jl")
include("group_utils.jl")
using NLsolve

@testset "Special Linear group" begin
    @testset "basic properties" begin
        G = SpecialLinear(3)
        @test G === SpecialLinear(3, ℝ)
        @test repr(G) == "SpecialLinear(3, ℝ)"
        @test base_manifold(G) === SpecialLinear(3)
        @test decorated_manifold(G) == GeneralLinear(3)
        @test number_system(G) === ℝ
        @test manifold_dimension(G) == 8
        @test representation_size(G) == (3, 3)
        Gc = SpecialLinear(2, ℂ)
        @test decorated_manifold(Gc) == GeneralLinear(2, ℂ)
        @test repr(Gc) == "SpecialLinear(2, ℂ)"
        @test number_system(Gc) == ℂ
        @test manifold_dimension(Gc) == 6
        @test representation_size(Gc) == (2, 2)
        Gh = SpecialLinear(4, ℍ)
        @test repr(Gh) == "SpecialLinear(4, ℍ)"
        @test number_system(Gh) == ℍ
        @test manifold_dimension(Gh) == 4 * 15
        @test representation_size(Gh) == (4, 4)

        @test (@inferred invariant_metric_dispatch(G, LeftAction())) === Val(true)
        @test (@inferred invariant_metric_dispatch(G, RightAction())) === Val(false)
        @test is_default_metric(
            MetricManifold(G, InvariantMetric(EuclideanMetric(), LeftAction())),
        ) === true
        @test @inferred(Manifolds.default_metric_dispatch(G, EuclideanMetric())) ===
              Val(true)
        @test @inferred(
            Manifolds.default_metric_dispatch(
                G,
                InvariantMetric(EuclideanMetric(), LeftAction()),
            )
        ) === Val(true)
        @test @inferred(
            Manifolds.default_metric_dispatch(
                MetricManifold(G, InvariantMetric(EuclideanMetric(), LeftAction())),
            )
        ) === Val(true)
        @test Manifolds.allocation_promotion_function(Gc, exp!, (1,)) === complex
    end

    @testset "Real" begin
        G = SpecialLinear(3)

        @test_throws DomainError is_point(G, randn(2, 3), true)
        @test_throws DomainError is_point(G, Float64[2 1; 1 1], true)
        @test_throws DomainError is_point(G, [1 0 im; im 0 0; 0 -1 0], true)
        @test_throws DomainError is_point(G, zeros(3, 3), true)
        @test_throws DomainError is_point(G, Float64[1 3 3; 1 1 2; 1 2 3], true)
        @test_throws DomainError is_point(
            G,
            make_identity(SpecialLinear(2), ones(2, 2)),
            true,
        )
        @test is_point(G, Float64[1 1 1; 2 2 1; 2 3 3], true)
        @test is_point(G, make_identity(G, ones(3, 3)), true)
        @test_throws DomainError is_tangent_vector(
            G,
            Float64[2 3 2; 3 1 2; 1 1 1],
            randn(3, 3),
            true;
            atol=1e-6,
        )
        @test_throws DomainError is_tangent_vector(
            G,
            Float64[2 1 2; 3 2 2; 2 2 1],
            Float64[2 1 -1; 2 2 1; 1 1 -1],
            true;
            atol=1e-6,
        )
        @test is_tangent_vector(
            G,
            Float64[2 1 2; 3 2 2; 2 2 1],
            Float64[-1 -1 -1; 1 -1 2; -1 -1 2],
            true;
            atol=1e-6,
        )

        types = [Matrix{Float64}]
        pts =
            [[2 -1 -3; 4 -1 -6; -1 1 2], [0 2 1; 0 -3 -1; 1 0 2], [-2 0 -1; 1 0 0; -1 -1 2]]
        vpts = [[0 -1 -5; 1 2 0; 1 2 -2], [0 -2 1; -2 1 2; -4 2 -1]]

        retraction_methods = [
            Manifolds.GroupExponentialRetraction(LeftAction()),
            Manifolds.GroupExponentialRetraction(RightAction()),
        ]

        inverse_retraction_methods = [
            Manifolds.GroupLogarithmicInverseRetraction(LeftAction()),
            Manifolds.GroupLogarithmicInverseRetraction(RightAction()),
        ]

        for T in types
            gpts = convert.(T, pts)
            vgpts = convert.(T, vpts)
            test_group(G, gpts, vgpts, vgpts; test_diff=true, test_invariance=true)
            test_manifold(
                G,
                gpts;
                test_reverse_diff=false,
                test_forward_diff=false,
                test_injectivity_radius=false,
                test_project_point=true,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                exp_log_atol_multiplier=1e10,
                retraction_atol_multiplier=1e8,
                is_tangent_atol_multiplier=1e10,
            )
        end

        @testset "project" begin
            p = randn(3, 3)
            @test !is_point(G, p)
            q = project(G, p)
            @test is_point(G, q)
            @test project(G, q) ≈ q

            X = randn(3, 3)
            @test !is_tangent_vector(G, q, X; atol=1e-6)
            Y = project(G, q, X)
            @test is_tangent_vector(G, q, Y; atol=1e-6)
            @test project(G, q, Y) ≈ Y
        end
    end

    @testset "Complex" begin
        G = SpecialLinear(2, ℂ)

        @test_throws DomainError is_point(G, randn(ComplexF64, 2, 3), true)
        @test_throws DomainError is_point(G, randn(2, 2), true)
        @test_throws DomainError is_point(G, ComplexF64[1 0 im; im 0 0; 0 -1 0], true)
        @test_throws DomainError is_point(G, ComplexF64[1 im; im 1], true)
        @test_throws DomainError is_point(
            G,
            make_identity(SpecialLinear(2), ones(ComplexF64, 2, 2)),
            true,
        )
        @test is_point(G, ComplexF64[im 1; -2 im], true)
        @test is_point(G, make_identity(G, ones(3, 3)), true)
        @test_throws DomainError is_tangent_vector(
            G,
            ComplexF64[-1+im -1; -im 1],
            ComplexF64[1-im 1+im; 1 -1+im],
            true;
            atol=1e-6,
        )
        @test_throws DomainError is_tangent_vector(
            G,
            ComplexF64[1 1+im; -1+im -1],
            ComplexF64[1-im -1-im; -im im],
            true;
            atol=1e-6,
        )
        @test is_tangent_vector(
            G,
            ComplexF64[1 1+im; -1+im -1],
            ComplexF64[1-im 1+im; 1 -1+im],
            true;
            atol=1e-6,
        )

        types = [Matrix{ComplexF64}]
        pts = [
            [0+1im 0+2im; 0+1im 0+1im],
            [-2+0im 0+1im; 0-1im -1+0im],
            [0+0im 0-1im; 0-1im -1+3im],
        ]
        vpts = [[0-1im -1+1im; -1+0im 0+1im], [1+1im 0+0im; 0+1im -1-1im]]

        retraction_methods = [
            Manifolds.GroupExponentialRetraction(LeftAction()),
            Manifolds.GroupExponentialRetraction(RightAction()),
        ]

        inverse_retraction_methods = [
            Manifolds.GroupLogarithmicInverseRetraction(LeftAction()),
            Manifolds.GroupLogarithmicInverseRetraction(RightAction()),
        ]

        for T in types
            gpts = convert.(T, pts)
            vgpts = convert.(T, vpts)
            test_group(G, gpts, vgpts, vgpts; test_diff=true, test_invariance=true)
            test_manifold(
                G,
                gpts;
                test_reverse_diff=false,
                test_forward_diff=false,
                test_injectivity_radius=false,
                test_project_point=true,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                exp_log_atol_multiplier=1e8,
                retraction_atol_multiplier=1e8,
                is_tangent_atol_multiplier=1e8,
            )
        end

        @testset "project" begin
            p = randn(ComplexF64, 2, 2)
            @test !is_point(G, p)
            q = project(G, p)
            @test is_point(G, q)
            @test project(G, q) ≈ q

            X = randn(ComplexF64, 2, 2)
            @test !is_tangent_vector(G, q, X; atol=1e-6)
            Y = project(G, q, X)
            @test is_tangent_vector(G, q, Y; atol=1e-6)
            @test project(G, q, Y) ≈ Y
        end
    end
end
