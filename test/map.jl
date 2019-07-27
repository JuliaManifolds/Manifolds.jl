
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

    @testset "ProductMap" begin
        O = Manifolds.Euclidean(1)
        M = Manifolds.Euclidean(2)
        N = Manifolds.Euclidean(3)
        PI = M × N
        PO = O × O

        f = Manifolds.FunctionMap(M, O, prod)
        g = Manifolds.FunctionMap(N, O, sum)
        h = f × g

        @test isa(h, Manifolds.ProductMap)
        @test Manifolds.domain(h) === PI
        @test Manifolds.codomain(h) === PO

        a, b = randn(2), randn(3)
        @test h(a, b) == (f(a), g(b))
    end

    @testset "IdentityMap" begin
        M = Manifolds.Euclidean(3)
        id = Manifolds.IdentityMap(M)
        @test Manifolds.domain(id) === M
        @test Manifolds.codomain(id) === M

        x = randn(3)
        @test id(x) === x

        f = Manifolds.FunctionMap(M, M, x -> x.*2)
        @test id ∘ id === id
        @test id ∘ f === f
        @test f ∘ id === f
    end

    @testset "Inclusion" begin
        M = Manifolds.Sphere(2)
        N = Manifolds.Euclidean(3)
        ϕ = Manifolds.Inclusion(M, N)
        @test Manifolds.domain(ϕ) === M
        @test Manifolds.codomain(ϕ) === N

        x = normalize(randn(3))
        @test ϕ(x) === x
    end

    @testset "RiemannianExponential" begin
        M = Manifolds.Sphere(2)
        x = normalize(randn(3))
        v = (I - x*x') * randn(3)
        f = Manifolds.Exp(M)
        g = (x, v) -> (x, exp(M, x, v))

        @test Manifolds.domain(f) === M × Euclidean(3)
        @test Manifolds.codomain(f) === M × M
        @test f(x, v) == g(x, v)
    end

    @testset "RiemannianLogarithm" begin
        M = Manifolds.Sphere(2)
        x = normalize(randn(3))
        y = normalize(randn(3))
        f = Manifolds.Log(M)
        g = (x, y) -> (x, log(M, x, y))

        @test Manifolds.domain(f) === M × M
        @test Manifolds.codomain(f) === M × Euclidean(3)
        @test f(x, y) == g(x, y)

        @test inv(f) === Manifolds.Exp(M)
        @test pinv(f) === Manifolds.Exp(M)
    end

    @testset "Geodesic" begin
        M = Manifolds.Sphere(2)
        x = normalize(randn(3))
        v = (I - x*x') * randn(3)
        f = Manifolds.Geodesic(M, x, v)
        g = geodesic(M, x, v)

        @test Manifolds.domain(f) === Euclidean(1)
        @test Manifolds.codomain(f) === M
        @test f(0) ≈ x
        @test ForwardDiff.derivative(f, 1e-10) ≈ v
        @test f(0.5) ≈ g(0.5)
    end

    @testset "ShortestGeodesic" begin
        M = Manifolds.Sphere(2)
        x = normalize(randn(3))
        y = normalize(randn(3))
        f = Manifolds.ShortestGeodesic(M, x, y)
        g = shortest_geodesic(M, x, y)

        @test Manifolds.domain(f) === Euclidean(1)
        @test Manifolds.codomain(f) === M
        @test f(0.0) ≈ x
        @test f(1.0) ≈ y
        @test f(0.5) ≈ g(0.5)
    end
end
