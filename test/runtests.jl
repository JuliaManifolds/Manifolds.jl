using ManifoldMuseum

using DoubleFloats
using ForwardDiff
using StaticArrays
using Test

"""
    test_manifold(m::Manifold, pts::AbstractVector)

Tests general properties of manifold `m`, given at least three different points
that lie on it (contained in `pts`).
"""
function test_manifold(M::Manifold, pts::AbstractVector)
    # log/exp
    length(pts) ≥ 3 || error("Not enough points (at least three expected)")
    isapprox(M, pts[1], pts[2]) && error("Points 1 and 2 are equal")
    isapprox(M, pts[1], pts[3]) && error("Points 1 and 3 are equal")

    tv1 = log(M, pts[1], pts[2])

    @testset "log/exp tests" begin
        @test isapprox(M, pts[2], exp(M, pts[1], tv1))
        @test isapprox(M, pts[1], exp(M, pts[1], tv1, 0))
        @test isapprox(M, pts[2], exp(M, pts[1], tv1, 1))
        for x ∈ pts
            @test isapprox(M, zero_tangent_vector(M, x), log(M, pts[1], pts[1]))
        end
        zero_tangent_vector!(M, tv1, pts[1])
        @test isapprox(M, pts[1], tv1, zero_tangent_vector(M, pts[1]))
        log!(M, tv1, pts[1], pts[2])
        @test norm(M, pts[1], tv1) ≈ sqrt(dot(M, pts[1], tv1, tv1))

        @test isapprox(M, exp(M, pts[1], tv1, 1), pts[2])
        @test isapprox(M, exp(M, pts[1], tv1, 0), pts[1])

        @test distance(M, pts[1], pts[2]) ≈ norm(M, pts[1], tv1)
    end

    @testset "linear algebra in tangent space" begin
        @test isapprox(M, pts[1], 0*tv1, zero_tangent_vector(M, pts[1]))
        @test isapprox(M, pts[1], 2*tv1, tv1+tv1)
        @test isapprox(M, pts[1], 0*tv1, tv1-tv1)
    end

    @testset "ForwardDiff support" begin
        exp_f(t) = distance(M, pts[1], exp(M, pts[1], t*tv1))
        d12 = distance(M, pts[1], pts[2])
        for t ∈ 0.1:0.1:1.0
            @test d12 ≈ ForwardDiff.derivative(exp_f, t)
        end
    end
end

function test_arraymanifold()
    M = ManifoldMuseum.Sphere(2)
    A = ArrayManifold(M)
    x = [1., 0., 0.]
    y = 1/sqrt(2)*[1., 1., 0.]
    z = [0., 1., 0.]
    v = log(M,x,y)
    v2 = log(A,x,y)
    y2 = exp(A,x,v2)
    w = log(M,x,z)
    w2 = log(A,x,z; atol=10^(-15))
    @test isapprox(y2.value,y)
    @test distance(A,x,y) == distance(M,x,y)
    @test norm(A,x,v) == norm(M,x,v)
    @test dot(A,x,v2,w2; atol=10^(-15)) == dot(M,x,v,w)
    @test_throws DomainError ManifoldMuseum.is_manifold_point(M,2*y)
    @test_throws DomainError ManifoldMuseum.is_tangent_vector(M,y,v; atol=10^(-15))

    test_manifold(A, [x, y, z])
end

@testset "Sphere" begin
    M = ManifoldMuseum.Sphere(2)
    types = [Vector{Float64},
             MVector{3, Float64},
             Vector{Float32},
             MVector{3, Float32},
             Vector{Double64},
             MVector{3, Double64}]
    for T in types
        test_manifold(M, [convert(T, [1.0, 0.0, 0.0]),
                          convert(T, [0.0, 1.0, 0.0]),
                          convert(T, [0.0, 0.0, 1.0])])
    end

    @testset "Distribution tests" begin
        usd_vector = ManifoldMuseum.uniform_distribution(M, [1.0, 0.0, 0.0])
        @test isa(rand(usd_vector), Vector)
        for _ in 1:10
            @test norm(rand(usd_vector)) ≈ 1.0
        end
        usd_mvector = ManifoldMuseum.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
        @test isa(rand(usd_mvector), MVector)

        x = [1.0, 0.0, 0.0]
        gtsd_vector = ManifoldMuseum.normal_tvector_distribution(M, x, 1.0)
        @test isa(rand(gtsd_vector), Vector)
        for _ in 1:10
            @test dot(x, rand(gtsd_vector)) ≈ 0.0
        end
        gtsd_mvector = ManifoldMuseum.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
        @test isa(rand(gtsd_mvector), MVector)
    end

    test_arraymanifold()
end
