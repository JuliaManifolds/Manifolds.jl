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
        q = fill(1.0 * im)
        X = fill(Complex(0.5))
        @test translate_diff(G, ig, q, X) === X

        @test identity_element(G) === 1.0
        @test identity_element(G, 1.0f0) === 1.0f0
        @test identity_element(G, fill(1.0f0)) == fill(1.0f0)
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

    @testset "array points" begin
        pts = [fill(1.0 + 0.0im), fill(0.0 + 1.0im), fill((1.0 + 1.0im) / √2)]
        Xpts = [fill(0.0 + 0.5im), fill(0.0 - 1.5im)]
        @test compose(G, pts[2], pts[1]) ≈ fill(pts[2] .* pts[1])
        @test translate_diff(G, pts[2], pts[1], Xpts[1]) ≈ fill(pts[2] .* Xpts[1])
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
            test_vector_spaces=false,
            test_project_tangent=true,
            test_musical_isomorphisms=false,
            test_default_vector_transport=true,
            is_mutating=false,
            exp_log_atol_multiplier=2.0,
            is_tangent_atol_multiplier=2.0,
            mid_point12=nothing,
        )

        @test isapprox(G, (1.0 + 1.0im) / √2, mean(G, pts))
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
        q = fill(0.0)
        X = fill(0.5)
        @test translate_diff(G, ig, q, X) === X

        @test identity_element(G) === 0.0
        @test identity_element(G, 1.0f0) === 0.0f0
        @test identity_element(G, fill(0.0f0)) == fill(0.0f0)
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

    @testset "array points" begin
        pts = [fill(1.0), fill(0.5), fill(-3.0)]
        Xpts = [fill(-2.0), fill(0.5), fill(2.0)]
        @test compose(G, pts[2], pts[1]) ≈ fill(pts[2] .+ pts[1])
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
            test_vector_spaces=false,
            test_project_tangent=true,
            test_musical_isomorphisms=false,
            test_default_vector_transport=true,
            is_mutating=false,
            exp_log_atol_multiplier=2.0,
            is_tangent_atol_multiplier=2.0,
            mid_point12=nothing,
        )
        @test isapprox(G, 0.5, mean(G, [1.0, 0.5, 0.0]))
    end
end
