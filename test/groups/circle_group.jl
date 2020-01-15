include("../utils.jl")
include("group_utils.jl")

@testset "Circle group" begin
    G = CircleGroup()
    @test repr(G) == "CircleGroup()"

    @test base_manifold(G) === Circle{ℂ}()

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
        test_group(G, pts, vpts; test_diff = true, test_mutating = false)
    end

    @testset "vector points" begin
        pts = [[1.0 + 0.0im], [0.0 + 1.0im], [(1.0 + 1.0im) / √2]]
        vpts = [[0.0 + 0.5im]]
        @test compose(G, pts[2], pts[1]) ≈ pts[2] .* pts[1]
        @test translate_diff(G, pts[2], pts[1], vpts[1]) ≈ pts[2] .* vpts[1]
        test_group(G, pts, vpts; test_diff = true, test_mutating = true)
    end
end
