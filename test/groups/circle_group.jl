include("../utils.jl")
include("group_utils.jl")

@testset "Circle group" begin
    G = CircleGroup()
    @test repr(G) == "CircleGroup()"

    @test base_manifold(G) === Circle{ℂ}()

    @test has_invariant_metric(G, LeftAction())
    @test has_invariant_metric(G, RightAction())
    @test has_biinvariant_metric(G)
    @test is_default_metric(MetricManifold(G, EuclideanMetric()))
    @test is_group_manifold(G)
    @testset "identity overloads" begin
        ig = Identity(G)
        @test inv(G, ig) === ig
        q = [1.0 * im]
        X = [Complex(0.5)]
        @test translate_diff(G, ig, q, X) === X

        @test identity_element(G) === 1.0
        @test identity_element(G, 1.0f0) === 1.0f0
        @test identity_element(G, [1.0f0]) == [1.0f0]
        @test !is_point(G, Identity(AdditionOperation()))
        ef = Identity(AdditionOperation())
        @test_throws DomainError is_point(G, ef, true)
        @test_throws DomainError is_vector(G, ef, X, true; check_base_point=true)
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
        pts = [1.0 + 0.0im, 0.0 + 1.0im, (1.0 + 1.0im) / √2]
        test_manifold(
            G,
            pts,
            test_forward_diff=false,
            test_reverse_diff=false,
            test_vector_spaces=false,
            test_project_tangent=true,
            test_musical_isomorphisms=false,
            test_default_vector_transport=true,
            is_mutating=false,
            exp_log_atol_multiplier=2.0,
            is_tangent_atol_multiplier=2.0,
            mid_point12=nothing,
        )
    end
end

@testset "Real circle group" begin
    G = RealCircleGroup()
    @test repr(G) == "RealCircleGroup()"

    @test base_manifold(G) === Circle{ℝ}()

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

    @testset "points" begin
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

    @testset "Group forwards" begin
        pts = [1.0, 0.5, -3.0]
        test_manifold(
            G,
            pts,
            test_forward_diff=false,
            test_reverse_diff=false,
            test_vector_spaces=false,
            test_project_tangent=true,
            test_musical_isomorphisms=false,
            test_default_vector_transport=true,
            is_mutating=false,
            exp_log_atol_multiplier=2.0,
            is_tangent_atol_multiplier=2.0,
            mid_point12=nothing,
        )
    end
end
