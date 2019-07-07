
@testset "Map" begin
    @testset "FunctionMap" begin
        M = Manifolds.Euclidean(3)
        N = Manifolds.Euclidean(3, 3)
        outer = x -> x*x'
        ϕ = Manifolds.FunctionMap(M, N, outer)
        @test Manifolds.domain(ϕ) === M
        @test Manifolds.codomain(ϕ) === N

        x = normalize(randn(3))
        @test ϕ(x) == outer(x)

        ψ = Manifolds.FunctionMap(M, N, ϕ)
        @test ψ === ψ
    end

    @testset "CompositeMap" begin
        M = Manifolds.Euclidean(3, 3)
        N = Manifolds.Euclidean(1)
        g = Manifolds.FunctionMap(M, M, x -> x.^2)
        f = Manifolds.FunctionMap(M, N, sum)
        h = f ∘ g
        @test isa(h, Manifolds.CompositeMap)
        @test Manifolds.domain(h) === M
        @test Manifolds.codomain(h) === N

        x = randn(3, 3)
        @test h(x) == sum(x.^2)
    end

    @testset "Injection" begin
        M = Manifolds.Sphere(2)
        N = Manifolds.Euclidean(3)
        f = x -> x .* 2
        linvf = x -> x ./ 2
        ϕ = Manifolds.Injection(M, N, f, linvf)
        @test Manifolds.domain(ϕ) === M
        @test Manifolds.codomain(ϕ) === N

        x = normalize(randn(3))
        @test ϕ(x) == f(x)
        @test (ϕ.linvf ∘ ϕ.f)(x) ≈ x
    end

    @testset "Surjection" begin
        M = Manifolds.Euclidean(3)
        N = Manifolds.Sphere(2)
        ϕ = Manifolds.Surjection(M, N, normalize, identity)
        @test Manifolds.domain(ϕ) === M
        @test Manifolds.codomain(ϕ) === N

        x = randn(3)
        @test ϕ(x) == normalize(x)

        ψ = pinv(ϕ)
        @test isa(ψ, Manifolds.Injection)
        @test Manifolds.domain(ψ) === N
        @test Manifolds.codomain(ψ) === M
        @test ψ(x) == x

        ω = ψ ∘ ϕ
        @test isa(ω, Manifolds.CompositeMap)
        @test Manifolds.domain(ω) === M
        @test Manifolds.codomain(ω) === M
        @test ω(x) == normalize(x)
    end

    @testset "Bijection" begin
        M = Manifolds.Rotations(2)
        N = Manifolds.Sphere(1)
        ϕ = Manifolds.Bijection(M, N, x -> x[1, 1:2], x -> [x[1] x[2]; -x[2] x[1]])
        @test Manifolds.domain(ϕ) === M
        @test Manifolds.codomain(ϕ) === N

        x = [0.0 1.0; -1.0 0.0]
        @test ϕ(x) == [0.0, 1.0]

        ϕinv = inv(ϕ)
        @test isa(ϕinv, Manifolds.Bijection)
        @test Manifolds.domain(ϕinv) === N
        @test Manifolds.codomain(ϕinv) === M

        y = ones(2) / sqrt(2)
        @test ϕinv(y) == [1 1; -1 1] ./ sqrt(2)

        ω = ϕinv ∘ ϕ
        @test isa(ω, Manifolds.Bijection)
        @test Manifolds.domain(ω) === M
        @test Manifolds.codomain(ω) === M
        @test ω(x) == x
        @test inv(ω)(x) == x
    end

    @testset "ExponentialMap" begin
        M = Manifolds.Sphere(2)
        x = normalize(randn(3))
        v = (I - x*x') * randn(3)
        f = Manifolds.Exponential(M, x)
        g = v -> exp(M, x, v)

        @test Manifolds.domain(f) === Euclidean(3)
        @test Manifolds.codomain(f) === M
        @test f(v) == g(v)
    end

    @testset "LogarithmMap" begin
        M = Manifolds.Sphere(2)
        x = normalize(randn(3))
        y = normalize(randn(3))
        f = Manifolds.Logarithm(M, x)
        g = y -> log(M, x, y)

        @test Manifolds.domain(f) === M
        @test Manifolds.codomain(f) === Euclidean(3)
        @test f(y) == g(y)
    end

    @testset "Geodesic" begin
        M = Manifolds.Sphere(2)
        x = normalize(randn(3))
        v = (I - x*x') * randn(3)
        f = Manifolds.Geodesic(M, x, v)
        g = geodesic(M, x, v)

        @test Manifolds.domain(f) === Euclidean(1)
        @test Manifolds.codomain(f) === M
        @test f(0) == x
        @test ForwardDiff.derivative(f, 1e-10) ≈ v
        @test f(0.5) == g(0.5)
    end

    @testset "ShortestGeodesic" begin
        M = Manifolds.Sphere(2)
        x = normalize(randn(3))
        y = normalize(randn(3))
        f = Manifolds.ShortestGeodesic(M, x, y)
        g = shortest_geodesic(M, x, y)

        @test Manifolds.domain(f) === Euclidean(1)
        @test Manifolds.codomain(f) === M
        @test f(0) == x
        @test f(1) == y
        @test f(0.5) == g(0.5)
    end
end
