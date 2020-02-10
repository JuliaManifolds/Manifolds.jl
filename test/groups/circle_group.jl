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

    @testset "identity overloads" begin
        @test identity(G, Identity(G)) === Identity(G)
        @test inv(G, Identity(G)) === Identity(G)
        y = [Complex(0.0)]
        @test identity!(G, y, [Complex(1.0)]) === y
        @test y == [Complex(1.0)]
        y = [1.0 * im]
        v = [Complex(0.5)]
        @test translate_diff(G, Identity(G), y, v) === v
    end

    @testset "scalar points" begin
        pts = [1.0 + 0.0im, 0.0 + 1.0im, (1.0 + 1.0im) / √2]
        vpts = [0.0 + 0.5im]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] * pts[1]
        @test translate_diff(G, pts[2], pts[1], vpts[1]) ≈ pts[2] * vpts[1]
        test_group(G, pts, vpts, vpts; test_diff = true, test_mutating = false, test_invariance = true)
    end

    @testset "vector points" begin
        pts = [[1.0 + 0.0im], [0.0 + 1.0im], [(1.0 + 1.0im) / √2]]
        vpts = [[0.0 + 0.5im]]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] .* pts[1]
        @test translate_diff(G, pts[2], pts[1], vpts[1]) ≈ pts[2] .* vpts[1]
        test_group(G, pts, vpts, vpts; test_diff = true, test_mutating = true, test_invariance = true)
    end

    @testset "Group forwards to decorated" begin
        pts = [[1.0 + 0.0im], [0.0 + 1.0im], [(1.0 + 1.0im) / √2]]
        test_manifold(
            G,
            pts,
            test_forward_diff = false,
            test_reverse_diff = false,
            test_vector_spaces = false,
            test_project_tangent = true,
            test_musical_isomorphisms = false,
            test_vector_transport = true,
            is_mutating = true,
            exp_log_atol_multiplier = 2.0,
            is_tangent_atol_multiplier = 2.0,
        )
    end
end
