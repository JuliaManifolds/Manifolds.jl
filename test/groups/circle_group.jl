include("../utils.jl")
include("group_utils.jl")

using Manifolds: invariant_metric_dispatch, default_metric_dispatch

@testset "Circle group" begin
    G = CircleGroup()
    @test repr(G) == "CircleGroup()"

    @test base_manifold(G) === Circle{ℂ}()

    @test (@inferred invariant_metric_dispatch(G, LeftAction())) === Val(true)
    @test (@inferred invariant_metric_dispatch(G, RightAction())) === Val(true)
    @test (@inferred Manifolds.biinvariant_metric_dispatch(G)) === Val(true)
    @test (@inferred default_metric_dispatch(MetricManifold(G, EuclideanMetric()))) ===
          Val(true)
    @test has_invariant_metric(G, LeftAction())
    @test has_invariant_metric(G, RightAction())
    @test has_biinvariant_metric(G)
    @test is_default_metric(MetricManifold(G, EuclideanMetric()))

    @testset "identity overloads" begin
        ig = Identity(G)
        @test inv(G, ig) === ig
        q = [1.0 * im]
        X = [Complex(0.5)]
        @test translate_diff(G, ig, q, X) === X

        @test identity_element(G) === 1.0
        @test identity_element(G, 1.0f0) === 1.0f0
        @test identity_element(G, [1.0f0]) == [1.0f0]
    end

    @testset "scalar points" begin
        pts = [1.0 + 0.0im, 0.0 + 1.0im, (1.0 + 1.0im) / √2]
        Xpts = [0.0 + 0.5im, 0.0 - 1.5im]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] * pts[1]
        @test translate_diff(G, pts[2], pts[1], Xpts[1]) ≈ pts[2] * Xpts[1]
        test_group(
            G,
            pts,
            Xpts,
            Xpts;
            test_diff=true,
            test_mutating=false,
            test_invariance=true,
            test_lie_bracket=true,
            test_adjoint_action=true,
        )
    end

    @testset "vector points" begin
        pts = [[1.0 + 0.0im], [0.0 + 1.0im], [(1.0 + 1.0im) / √2]]
        Xpts = [[0.0 + 0.5im], [0.0 - 1.5im]]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] .* pts[1]
        @test translate_diff(G, pts[2], pts[1], Xpts[1]) ≈ pts[2] .* Xpts[1]
        test_group(
            G,
            pts,
            Xpts,
            Xpts;
            test_diff=true,
            test_mutating=true,
            test_invariance=true,
            test_lie_bracket=true,
            test_adjoint_action=true,
        )
    end

    @testset "Group forwards to decorated" begin
        pts = [[1.0 + 0.0im], [0.0 + 1.0im], [(1.0 + 1.0im) / √2]]
        test_manifold(
            G,
            pts,
            basis_types_to_from=(Manifolds.VeeOrthogonalBasis(), DefaultOrthonormalBasis()),
            test_forward_diff=false,
            test_reverse_diff=false,
            test_vector_spaces=false,
            test_project_tangent=true,
            test_musical_isomorphisms=false,
            test_default_vector_transport=true,
            is_mutating=true,
            exp_log_atol_multiplier=2.0,
            is_tangent_atol_multiplier=2.0,
        )
    end
end

@testset "Real circle group" begin
    G = RealCircleGroup()
    @test repr(G) == "RealCircleGroup()"

    @test base_manifold(G) === Circle{ℝ}()

    @test (@inferred invariant_metric_dispatch(G, LeftAction())) === Val(true)
    @test (@inferred invariant_metric_dispatch(G, RightAction())) === Val(true)
    @test (@inferred Manifolds.biinvariant_metric_dispatch(G)) === Val(true)
    @test (@inferred default_metric_dispatch(MetricManifold(G, EuclideanMetric()))) ===
          Val(true)
    @test has_invariant_metric(G, LeftAction())
    @test has_invariant_metric(G, RightAction())
    @test has_biinvariant_metric(G)
    @test is_default_metric(MetricManifold(G, EuclideanMetric()))

    @testset "identity overloads" begin
        ig = Identity(G)
        @test inv(G, ig) === ig
        q = [0.0]
        X = [0.5]
        @test translate_diff(G, ig, q, X) === X

        @test identity_element(G) === 0.0
        @test identity_element(G, 1.0f0) === 0.0f0
        @test identity_element(G, [0.0f0]) == [0.0f0]
    end

    @testset "scalar points" begin
        pts = [1.0, 0.5, -3.0]
        Xpts = [-2.0, 0.5, 2.0]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] + pts[1]
        @test translate_diff(G, pts[2], pts[1], Xpts[1]) ≈ Xpts[1]
        test_group(
            G,
            pts,
            Xpts,
            Xpts;
            test_diff=true,
            test_mutating=false,
            test_invariance=true,
            test_lie_bracket=true,
            test_adjoint_action=true,
        )
    end

    @testset "vector points" begin
        pts = [[1.0], [0.5], [-3.0]]
        Xpts = [[-2.0], [0.5], [2.0]]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] .+ pts[1]
        @test translate_diff(G, pts[2], pts[1], Xpts[1]) ≈ Xpts[1]
        test_group(
            G,
            pts,
            Xpts,
            Xpts;
            test_diff=true,
            test_mutating=true,
            test_invariance=true,
            test_lie_bracket=true,
            test_adjoint_action=true,
        )
    end

    @testset "Group forwards to decorated" begin
        pts = [[1.0], [0.5], [-3.0]]
        test_manifold(
            G,
            pts,
            basis_types_to_from=(Manifolds.VeeOrthogonalBasis(), DefaultOrthonormalBasis()),
            test_forward_diff=true,
            test_reverse_diff=false,
            test_vector_spaces=false,
            test_project_tangent=true,
            test_musical_isomorphisms=false,
            test_default_vector_transport=true,
            is_mutating=true,
            exp_log_atol_multiplier=2.0,
            is_tangent_atol_multiplier=2.0,
        )
    end
end
