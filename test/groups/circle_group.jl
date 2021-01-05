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
        ig = Identity(G, [Complex(1.0)])
        @test identity(G, ig) === ig
        @test inv(G, ig) === ig
        @test allocate_result(G, get_coordinates, ig, Complex(1.0), DefaultBasis()) isa
              Array{Complex{Float64},1}
        y = [Complex(0.0)]
        @test identity!(G, y, [Complex(1.0)]) === y
        @test y == [Complex(1.0)]
        y = [1.0 * im]
        v = [Complex(0.5)]
        @test translate_diff(G, ig, y, v) === v
    end

    @testset "scalar points" begin
        pts = [1.0 + 0.0im, 0.0 + 1.0im, (1.0 + 1.0im) / √2]
        vpts = [0.0 + 0.5im]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] * pts[1]
        @test translate_diff(G, pts[2], pts[1], vpts[1]) ≈ pts[2] * vpts[1]
        test_group(
            G,
            pts,
            vpts,
            vpts;
            test_diff=true,
            test_mutating=false,
            test_invariance=true,
        )
    end

    @testset "vector points" begin
        pts = [[1.0 + 0.0im], [0.0 + 1.0im], [(1.0 + 1.0im) / √2]]
        vpts = [[0.0 + 0.5im]]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] .* pts[1]
        @test translate_diff(G, pts[2], pts[1], vpts[1]) ≈ pts[2] .* vpts[1]
        test_group(
            G,
            pts,
            vpts,
            vpts;
            test_diff=true,
            test_mutating=true,
            test_invariance=true,
        )
    end

    @testset "Group forwards to decorated" begin
        pts = [[1.0 + 0.0im], [0.0 + 1.0im], [(1.0 + 1.0im) / √2]]
        test_manifold(
            G,
            pts,
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
