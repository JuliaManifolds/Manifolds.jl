using ManifoldMuseum

using LinearAlgebra
using DoubleFloats
using ForwardDiff
using StaticArrays
using SimpleTraits
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
        @test norm(M, pts[1], tv1) ≈ sqrt(inner(M, pts[1], tv1, tv1))

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
    @test inner(A,x,v2,w2; atol=10^(-15)) == inner(M,x,v,w)
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

@testset "Metric" begin
    @testset "Scaled Euclidean" begin
        struct TestEuclidean{N} <: Manifold end
        struct TestEuclideanMetric <: Metric end
        ManifoldMuseum.manifold_dimension(::TestEuclidean{N}) where {N} = N
        function ManifoldMuseum.local_metric(M::MetricManifold{<:TestEuclidean,<:TestEuclideanMetric}, x)
            return Diagonal(1.0:manifold_dimension(M))
        end

        n = 3
        E = TestEuclidean{n}()
        g = TestEuclideanMetric()
        M = MetricManifold(E, g)
        G = Diagonal(1.0:n)
        invG = inv(G)
        @test manifold_dimension(M) == n
        @test base_manifold(M) === E
        @test metric(M) === g

        for vtype in (Vector, SVector{n}, MVector{n})
            x, v, w = vtype(randn(n)), vtype(randn(n)), vtype(randn(n))

            @test local_metric(M, x) ≈ G
            @test inverse_local_metric(M, x) ≈ invG
            @test det_local_metric(M, x) ≈ *(1.0:n...)
            @test inner(M, x, v, w) ≈ dot(v, G * w)
            @test norm(M, x, v) ≈ sqrt(dot(v, G * v))

            T = 0:.5:10
            @test exp(M, x, v, T) ≈ [x + t * v for t in T]
            @test geodesic(M, x, v, T) ≈ [x + t * v for t in T]

            @test christoffel_symbols_first(M, x) ≈ zeros(n, n, n)
            @test christoffel_symbols_second(M, x) ≈ zeros(n, n, n)
            @test riemann_tensor(M, x) ≈ zeros(n, n, n, n)
            @test ricci_tensor(M, x) ≈ zeros(n, n)
            @test ricci_curvature(M, x) ≈ 0
            @test gaussian_curvature(M, x) ≈ 0
            @test einstein_tensor(M, x) ≈ zeros(n, n)
        end
    end

    @testset "Scaled Sphere" begin
        struct TestSphere{N,T} <: Manifold
            r::T
        end
        struct TestSphericalMetric <: Metric end
        ManifoldMuseum.manifold_dimension(::TestSphere{N}) where {N} = N
        function ManifoldMuseum.local_metric(M::MetricManifold{<:TestSphere,<:TestSphericalMetric}, x)
            r = base_manifold(M).r
            d = similar(x)
            d[1] = r^2
            d[2] = d[1] * sin(x[1])^2
            return Diagonal(d)
        end
        sph_to_cart(θ, ϕ) = [cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)]

        n = 2
        r = 10 * rand()
        θ, ϕ = π * rand(), 2π * rand()
        Sr = TestSphere{n,Float64}(r)
        S = ManifoldMuseum.Sphere(n)
        g = TestSphericalMetric()
        M = MetricManifold(Sr, g)

        @test manifold_dimension(M) == n
        @test base_manifold(M) === Sr
        @test metric(M) === g

        for vtype in (Vector, SVector{n}, MVector{n})
            x = vtype([θ, ϕ])
            G = Diagonal(vtype([1, sin(θ)^2])) .* r^2
            invG = Diagonal(vtype([1, 1 / sin(θ)^2])) ./ r^2
            v, w = normalize(randn(n)), normalize(randn(n))

            @test local_metric(M, x) ≈ G
            @test inverse_local_metric(M, x) ≈ invG
            @test det_local_metric(M, x) ≈ r^4 * sin(θ)^2
            @test inner(M, x, v, w) ≈ dot(v, G * w)
            @test norm(M, x, v) ≈ sqrt(dot(v, G * v))

            xcart = sph_to_cart(θ, ϕ)
            vcart = [cos(ϕ)*cos(θ) -sin(ϕ)*sin(θ);
                     sin(ϕ)*cos(θ)  cos(ϕ)*sin(θ);
                     -sin(θ) 0] * v

            T = 0:.1:1
            @test isapprox([sph_to_cart(yi...) for yi in exp(M, x, v, T)],
                           exp(S, xcart, vcart, T); atol=1e-3, rtol=1e-3)
            @test isapprox([sph_to_cart(yi...) for yi in geodesic(M, x, v, T)],
                           geodesic(S, xcart, vcart, T); atol=1e-3, rtol=1e-3)

            Γ₁ = christoffel_symbols_first(M, x)
            for i=1:n, j=1:n, k=1:n
                if (i,j,k) == (1,2,2) || (i,j,k) == (2,1,2)
                    @test Γ₁[i,j,k] ≈ r^2*cos(θ)*sin(θ)
                elseif (i,j,k) == (2,2,1)
                    @test Γ₁[i,j,k] ≈ -r^2*cos(θ)*sin(θ)
                else
                    @test Γ₁[i,j,k] ≈ 0
                end
            end

            Γ₂ = christoffel_symbols_second(M, x)
            for i=1:n, j=1:n, k=1:n
                if (i,j,k) == (2,2,1)
                    @test Γ₂[i,j,k] ≈ -cos(θ)*sin(θ)
                elseif (i,j,k) == (1,2,2) || (i,j,k) == (2,1,2)
                    @test Γ₂[i,j,k] ≈ cot(θ)
                else
                    @test Γ₂[i,j,k] ≈ 0
                end
            end

            R = riemann_tensor(M, x)
            for i=1:n, j=1:n, k=1:n, l=1:n
                if (i,j,k,l) == (1,1,2,2)
                    @test R[i,j,k,l] ≈ -1
                elseif (i,j,k,l) == (1,2,1,2)
                    @test R[i,j,k,l] ≈ 1
                elseif (i,j,k,l) == (2,1,2,1)
                    @test R[i,j,k,l] ≈ sin(θ)^2
                elseif (i,j,k,l) == (2,2,1,1)
                    @test R[i,j,k,l] ≈ -sin(θ)^2
                else
                    @test R[i,j,k,l] ≈ 0
                end
            end

            @test ricci_tensor(M, x) ≈ G ./ r^2
            @test ricci_curvature(M, x) ≈ 2 / r^2
            @test gaussian_curvature(M, x) ≈ 1 / r^2
            @test einstein_tensor(M, x) ≈ ricci_tensor(M, x) - gaussian_curvature(M, x)  .* G
        end
    end

    @testset "Has Metric" begin
        struct BaseManifold{N} <: Manifold end
        struct BaseManifoldMetric{M} <: Metric end
        ManifoldMuseum.manifold_dimension(::BaseManifold{N}) where {N} = N
        @traitimpl HasMetric{BaseManifold,BaseManifoldMetric}
        ManifoldMuseum.inner(::BaseManifold, x, v, w) = 2 * dot(v,w)
        ManifoldMuseum.exp!(::BaseManifold, y, x, v) = y .= x + 2 * v
        ManifoldMuseum.log!(::BaseManifold, v, x, y) = v .= (y - x) / 2

        M = BaseManifold{3}()
        g = BaseManifoldMetric{3}()
        MM = MetricManifold(M, g)
        x = randn(3)
        v = randn(3)
        w = randn(3)
        y = similar(x)

        @test inner(M, x, v, w) == 2 * dot(v,w)
        @test inner(MM, x, v, w) === inner(M, x, v, w)
        @test exp(M, x, v) == x + 2 * v
        @test exp!(MM, y, x, v) === exp!(M, y, x, v)
        @test log(M, x, y) == (y - x) / 2
        @test log!(MM, v, x, y) === log!(M, v, x, y)
    end
end
