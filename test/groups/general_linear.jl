include("../utils.jl")
include("group_utils.jl")
using NLsolve

@testset "General Linear group" begin
    @testset "basic properties" begin
        G = GeneralLinear(3)
        @test G === GeneralLinear(3, ℝ)
        @test repr(G) == "GeneralLinear(3, ℝ)"
        @test base_manifold(G) === GeneralLinear(3)
        @test decorated_manifold(G) === Euclidean(3, 3)
        @test number_system(G) === ℝ
        @test manifold_dimension(G) == 9
        @test representation_size(G) == (3, 3)
        Gc = GeneralLinear(2, ℂ)
        @test decorated_manifold(Gc) === Euclidean(2, 2; field=ℂ)
        @test repr(Gc) == "GeneralLinear(2, ℂ)"
        @test number_system(Gc) == ℂ
        @test manifold_dimension(Gc) == 8
        @test representation_size(Gc) == (2, 2)
        Gh = GeneralLinear(4, ℍ)
        @test repr(Gh) == "GeneralLinear(4, ℍ)"
        @test number_system(Gh) == ℍ
        @test manifold_dimension(Gh) == 4 * 16
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

    @testset "GL(1,𝔽) special cases" begin
        @testset "real" begin
            G = GeneralLinear(1)
            p = 3.0 * ones(1, 1)
            X = 1.0 * ones(1, 1)
            @test exp(G, p, X) ≈ p * exp(X)' * exp(X - X')
            q = exp(G, p, X)
            Y = log(G, p, q)
            @test Y ≈ X
            @test exp_lie(G, X) ≈ exp(X)
            @test log_lie(G, exp(X)) ≈ X
            @test log_lie(G, [1.0]) == zeros(1) # vector to vector
            log_lie(G, Identity(G)) == zeros(1, 1) # Matrix to matrix
        end
        @testset "complex" begin
            G = GeneralLinear(1, ℂ)
            p = (1 + im) * ones(1, 1)
            X = (1 - im) * ones(1, 1)
            @test exp(G, p, X) ≈ p * exp(X)' * exp(X - X')
            q = exp(G, p, X)
            Y = log(G, p, q)
            @test Y ≈ X
            @test exp_lie(G, X) ≈ exp(X)
            @test log_lie(G, exp(X)) ≈ X
        end
    end

    @testset "Real" begin
        G = GeneralLinear(3)

        @test_throws DomainError is_point(G, randn(2, 3), true)
        @test_throws DomainError is_point(G, randn(2, 2), true)
        @test_throws DomainError is_point(G, randn(ComplexF64, 3, 3), true)
        @test_throws DomainError is_point(G, zeros(3, 3), true)
        @test_throws DomainError is_point(G, Float64[0 0 0; 0 1 1; 1 1 1], true)
        @test is_point(G, Float64[0 0 1; 0 1 1; 1 1 1], true)
        @test is_point(G, Identity(G), true)
        @test_throws DomainError is_vector(
            G,
            Float64[0 1 1; 0 1 1; 1 0 0],
            randn(3, 3),
            true,
        )
        @test is_vector(G, Float64[0 0 1; 0 1 1; 1 1 1], randn(3, 3), true)

        types = [Matrix{Float64}]
        pts = [
            Matrix(Diagonal([1, 2, 3])),
            [-2 5 -5; 0 2 -1; -3 -5 -2],
            [-5 1 0; 1 0 1; 0 1 3],
        ]
        vpts = [[-1 -2 0; -2 1 -2; 2 0 2], [1 1 1; 0 0 -2; 2 0 0]]

        retraction_methods = [
            Manifolds.GroupExponentialRetraction(LeftAction()),
            Manifolds.GroupExponentialRetraction(RightAction()),
        ]

        inverse_retraction_methods = [
            Manifolds.GroupLogarithmicInverseRetraction(LeftAction()),
            Manifolds.GroupLogarithmicInverseRetraction(RightAction()),
        ]

        basis_types = [DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd)]

        for T in types
            gpts = convert.(T, pts)
            vgpts = convert.(T, vpts)
            test_group(
                G,
                gpts,
                vgpts,
                vgpts;
                test_diff=true,
                test_invariance=true,
                test_lie_bracket=true,
                test_adjoint_action=true,
            )
            test_manifold(
                G,
                gpts;
                test_reverse_diff=false,
                test_forward_diff=false,
                test_project_point=true,
                test_injectivity_radius=false,
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
                basis_types_vecs=basis_types,
                basis_types_to_from=basis_types,
                exp_log_atol_multiplier=1e7,
                retraction_atol_multiplier=1e7,
            )
        end
    end

    @testset "Complex" begin
        G = GeneralLinear(2, ℂ)

        @test_throws DomainError is_point(G, randn(ComplexF64, 2, 3), true)
        @test_throws DomainError is_point(G, randn(ComplexF64, 3, 3), true)
        @test_throws DomainError is_point(G, zeros(2, 2), true)
        @test_throws DomainError is_point(G, ComplexF64[1 im; 1 im], true)
        @test is_point(G, ComplexF64[1 1; im 1], true)
        @test is_point(G, Identity(G), true)
        @test_throws DomainError is_point(G, Float64[0 0 0; 0 1 1; 1 1 1], true)
        @test_throws DomainError is_vector(
            G,
            ComplexF64[im im; im im],
            randn(ComplexF64, 2, 2),
            true,
        )
        @test is_vector(G, ComplexF64[1 im; im im], randn(ComplexF64, 2, 2), true)

        types = [Matrix{ComplexF64}]
        pts = [
            [-1-5im -1+3im; -6-4im 4+6im],
            [1+3im -1-4im; -2-2im -3-1im],
            [-6+0im 1+1im; 1-1im -4+0im],
        ]
        vpts = [[1+0im -2-1im; -1-2im -4+1im], [-2+2im -1-1im; -1-1im -3+0im]]

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
                test_project_point=true,
                test_injectivity_radius=false,
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
            )
        end
    end
end
