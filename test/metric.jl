using FiniteDifferences, ForwardDiff, OrdinaryDiffEq
using LinearAlgebra: I
using StatsBase: AbstractWeights, pweights
import Manifolds: mean!, median!

include("utils.jl")

struct TestEuclidean{N} <: Manifold end
struct TestEuclideanMetric <: Metric end

@testset "scaled Euclidean metric" begin
    Manifolds.manifold_dimension(::TestEuclidean{N}) where {N} = N
    function Manifolds.local_metric(M::MetricManifold{<:TestEuclidean,<:TestEuclideanMetric}, x)
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

    @test_throws MethodError local_metric_jacobian(E, zeros(3))
    @test_throws MethodError christoffel_symbols_second_jacobian(E, zeros(3))

    for vtype in (Vector, SVector{n}, MVector{n})
        x, v, w = vtype(randn(n)), vtype(randn(n)), vtype(randn(n))

        @test local_metric(M, x) ≈ G
        @test inverse_local_metric(M, x) ≈ invG
        @test det_local_metric(M, x) ≈ *(1.0:n...)
        @test log_local_metric_density(M, x) ≈ sum(log.(1.0:n)) / 2
        @test inner(M, x, v, w) ≈ dot(v, G * w)
        @test norm(M, x, v) ≈ sqrt(dot(v, G * v))

        T = 0:.5:10
        @test exp(M, x, v, T) ≈ [x + t * v for t in T]
        @test geodesic(M, x, v, T) ≈ [x + t * v for t in T]

        @test christoffel_symbols_first(M, x) ≈ zeros(n, n, n) atol=1e-6
        @test christoffel_symbols_second(M, x) ≈ zeros(n, n, n) atol=1e-6
        @test riemann_tensor(M, x) ≈ zeros(n, n, n, n) atol=1e-6
        @test ricci_tensor(M, x) ≈ zeros(n, n) atol=1e-6
        @test ricci_curvature(M, x) ≈ 0 atol=1e-6
        @test gaussian_curvature(M, x) ≈ 0 atol=1e-6
        @test einstein_tensor(M, x) ≈ zeros(n, n) atol=1e-6

        fdm = forward_fdm(2, 1)
        @test christoffel_symbols_first(M, x; backend=fdm) ≈ zeros(n, n, n) atol=1e-6
        @test christoffel_symbols_second(M, x; backend=fdm) ≈ zeros(n, n, n) atol=1e-6
        @test riemann_tensor(M, x; backend=fdm) ≈ zeros(n, n, n, n) atol=1e-6
        @test ricci_tensor(M, x; backend=fdm) ≈ zeros(n, n) atol=1e-6
        @test ricci_curvature(M, x; backend=fdm) ≈ 0 atol=1e-6
        @test gaussian_curvature(M, x; backend=fdm) ≈ 0 atol=1e-6
        @test einstein_tensor(M, x; backend=fdm) ≈ zeros(n, n) atol=1e-6

        @test christoffel_symbols_first(M, x; backend=:forwarddiff) ≈ zeros(n, n, n) atol=1e-6
        @test christoffel_symbols_second(M, x; backend=:forwarddiff) ≈ zeros(n, n, n) atol=1e-6
        @test riemann_tensor(M, x; backend=:forwarddiff) ≈ zeros(n, n, n, n) atol=1e-6
        @test ricci_tensor(M, x; backend=:forwarddiff) ≈ zeros(n, n) atol=1e-6
        @test ricci_curvature(M, x; backend=:forwarddiff) ≈ 0 atol=1e-6
        @test gaussian_curvature(M, x; backend=:forwarddiff) ≈ 0 atol=1e-6
        @test einstein_tensor(M, x; backend=:forwarddiff) ≈ zeros(n, n) atol=1e-6
    end
end

struct TestSphere{N,T} <: Manifold
    r::T
end

struct TestSphericalMetric <: Metric end

@testset "scaled Sphere metric" begin
    Manifolds.manifold_dimension(::TestSphere{N}) where {N} = N
    function Manifolds.local_metric(M::MetricManifold{<:TestSphere,<:TestSphericalMetric}, x)
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
    S = Manifolds.Sphere(n)
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
        @test log_local_metric_density(M, x) ≈ 2log(r) + log(sin(θ))
        @test inner(M, x, v, w) ≈ dot(v, G * w)
        @test norm(M, x, v) ≈ sqrt(dot(v, G * v))

        xcart = sph_to_cart(θ, ϕ)
        vcart = [cos(ϕ)*cos(θ) -sin(ϕ)*sin(θ);
                 sin(ϕ)*cos(θ)  cos(ϕ)*sin(θ);
                 -sin(θ) 0] * v

        if !Sys.iswindows() || Sys.ARCH == :x86_64
            @testset "numerically integrated geodesics for $vtype" begin
                T = 0:.1:1
                @test isapprox([sph_to_cart(yi...) for yi in exp(M, x, v, T)],
                               exp(S, xcart, vcart, T); atol=1e-3, rtol=1e-3)
                @test isapprox([sph_to_cart(yi...) for yi in geodesic(M, x, v, T)],
                               geodesic(S, xcart, vcart, T); atol=1e-3, rtol=1e-3)
            end
        end

        Γ₁ = christoffel_symbols_first(M, x)
        for i=1:n, j=1:n, k=1:n
            if (i,j,k) == (1,2,2) || (i,j,k) == (2,1,2)
                @test Γ₁[i,j,k] ≈ r^2*cos(θ)*sin(θ)
            elseif (i,j,k) == (2,2,1)
                @test Γ₁[i,j,k] ≈ -r^2*cos(θ)*sin(θ)
            else
                @test Γ₁[i,j,k] ≈ 0 atol=1e-6
            end
        end

        Γ₂ = christoffel_symbols_second(M, x)
        for l=1:n, i=1:n, j=1:n
            if (l,i,j) == (1,2,2)
                @test Γ₂[l,i,j] ≈ -cos(θ)*sin(θ)
            elseif (l,i,j) == (2,1,2) || (l,i,j) == (2,2,1)
                @test Γ₂[l,i,j] ≈ cot(θ)
            else
                @test Γ₂[l,i,j] ≈ 0 atol=1e-6
            end
        end

        R = riemann_tensor(M, x)
        for l=1:n, i=1:n, j=1:n, k=1:n
            if (l,i,j,k) == (2,1,1,2)
                @test R[l,i,j,k] ≈ -1
            elseif (l,i,j,k) == (2,1,2,1)
                @test R[l,i,j,k] ≈ 1
            elseif (l,i,j,k) == (1,2,1,2)
                @test R[l,i,j,k] ≈ sin(θ)^2
            elseif (l,i,j,k) == (1,2,2,1)
                @test R[l,i,j,k] ≈ -sin(θ)^2
            else
                @test R[l,i,j,k] ≈ 0 atol=1e-6
            end
        end

        @test ricci_tensor(M, x) ≈ G ./ r^2
        @test ricci_curvature(M, x) ≈ 2 / r^2
        @test gaussian_curvature(M, x) ≈ 1 / r^2
        @test einstein_tensor(M, x) ≈ ricci_tensor(M, x) - gaussian_curvature(M, x)  .* G
    end
end

struct BaseManifold{N} <: Manifold end
struct BaseManifoldMetric{M} <: Metric end
struct DefaultBaseManifoldMetric <: Metric end

@testset "Metric decorator" begin
    Manifolds.manifold_dimension(::BaseManifold{N}) where {N} = N
    Manifolds.inner(::BaseManifold, x, v, w) = 2 * dot(v,w)
    Manifolds.exp!(::BaseManifold, y, x, v) = y .= x + 2 * v
    Manifolds.log!(::BaseManifold, v, x, y) = v .= (y - x) / 2
    Manifolds.project_tangent!(::BaseManifold, w, x, v) = w .= 2 .* v
    Manifolds.project_point!(::BaseManifold, y, x) = (y .= x)
    Manifolds.injectivity_radius(::BaseManifold) = Inf
    Manifolds.local_metric(::MetricManifold{BaseManifold{N},BaseManifoldMetric{N}},x) where N = 2*one(x*x')
    Manifolds.exp!(::MetricManifold{BaseManifold{N},BaseManifoldMetric{N}}, y, x, v) where N = exp!(base_manifold(M), y, x, v)
    Manifolds.vector_transport_to!(::BaseManifold, vto, x, v, y, ::ParallelTransport) = (vto .= v)
    Manifolds.is_default_metric(M::BaseManifold,G::DefaultBaseManifoldMetric) = Val(true)
    Manifolds.tangent_orthonormal_basis(M::BaseManifold{N},x,v) where {N} = ( [(Matrix(I, N, N)[:,i]) for i in 1:N], zeros(N))
    Manifolds.projected_distribution(M::BaseManifold, d) = ProjectedPointDistribution(M, d, project_point!, rand(d))
    Manifolds.projected_distribution(M::BaseManifold, d, x) = ProjectedPointDistribution(M, d, project_point!, x)
    Manifolds.mean!(M::BaseManifold, y, x::AbstractVector, w::AbstractVector; kwargs...) = fill!(y, 1)
    Manifolds.median!(M::BaseManifold, y, x::AbstractVector, w::AbstractVector; kwargs...) = fill!(y, 2)
    Manifolds.mean!(::MetricManifold{BaseManifold{N},BaseManifoldMetric{N}}, y, x::AbstractVector, w::AbstractVector; kwargs...) where {N} = fill!(y, 3)
    Manifolds.median!(::MetricManifold{BaseManifold{N},BaseManifoldMetric{N}}, y, x::AbstractVector, w::AbstractVector; kwargs...) where {N} = fill!(y, 4)

    function Manifolds.flat!(::BaseManifold, v::FVector{Manifolds.CotangentSpaceType}, x, w::FVector{Manifolds.TangentSpaceType})
        v.data .= 2 .* w.data
        return v
    end
    function Manifolds.sharp!(::BaseManifold, v::FVector{Manifolds.TangentSpaceType}, x, w::FVector{Manifolds.CotangentSpaceType})
        v.data .= w.data ./ 2
        return v
    end

    M = BaseManifold{3}()
    g = BaseManifoldMetric{3}()
    MM = MetricManifold(M, g)
    g2 = DefaultBaseManifoldMetric()
    MM2 = MetricManifold(M,g2)

    @test is_decorator_manifold(MM) == Val{true}()
    @test is_decorator_manifold(MM2) == Val{true}()
    @test is_decorator_manifold(M) == Val{false}()
    @test is_default_metric(MM) == is_default_metric(base_manifold(MM),metric(MM))
    @test is_default_metric(MM2) == is_default_metric(base_manifold(MM2),metric(MM2))
    @test is_default_metric(MM2) == Val{true}()

    @test convert(typeof(MM2),M) == MM2
    @test_throws ErrorException convert(typeof(MM),M)
    x = [0.1 0.2 0.4]
    v = [0.5 0.7 0.11]
    w = [0.13 0.17 0.19]
    y = similar(x)

    # Test fallbacks
    @test_throws ErrorException vee!(M,w,x,v)
    @test_throws ErrorException hat!(M,w,x,v)

    @test inner(M, x, v, w) == 2 * dot(v,w)
    @test inner(MM, x, v, w) === inner(M, x, v, w)
    @test norm(MM, x, v) === norm(M, x, v)
    @test exp(M, x, v) == x + 2 * v
    @test exp(MM2, x, v) == exp(M, x, v)
    @test exp!(MM, y, x, v) === exp!(M, y, x, v)
    @test retract!(MM, y, x, v) === retract!(M, y, x, v)
    @test retract!(MM, y, x, v, 1) === retract!(M, y, x, v, 1)
    # without a definition for the metric from the embedding, no projection possible
    @test_throws ErrorException log!(MM, w, x, y) === project_tangent!(M, w, x, y)
    @test_throws ErrorException project_tangent!(MM, w, x, v) === project_tangent!(M, w, x, v)
    @test_throws ErrorException project_point!(MM, y, x) === project_point!(M, y, x)
    @test_throws ErrorException vector_transport_to!(MM, w, x, v, y) === vector_transport_to!(M, w, x, v, y)
    # without DiffEq, these error
    # @test_throws ErrorException exp(MM,x, v, 1:3)
    # @test_throws ErrorException exp!(MM, y, x, v)
    # these always fall back anyways.
    @test zero_tangent_vector!(MM, v, x) === zero_tangent_vector!(M, v, x)
    @test injectivity_radius(MM, x) === injectivity_radius(M, x)
    @test injectivity_radius(MM) === injectivity_radius(M)
    @test is_manifold_point(MM, x) === is_manifold_point(M, x)
    @test is_tangent_vector(MM, x, v) === is_tangent_vector(M, x, v)

    @test_throws ErrorException local_metric(MM2,x)
    @test_throws ErrorException local_metric_jacobian(MM2,x)
    @test_throws ErrorException christoffel_symbols_second_jacobian(MM2,x)
    # MM falls back to nondefault error
    @test_throws ErrorException projected_distribution(MM,1,x)
    @test_throws ErrorException projected_distribution(MM,1)
    @test_throws ErrorException normal_tvector_distribution(MM,x,0.2)

    @test inner(MM2, x, v, w) === inner(M, x, v, w)
    @test norm(MM2, x, v) === norm(M, x, v)
    @test distance(MM2, x, y) === distance(M, x, y)
    @test exp!(MM2, y, x, v) === exp!(M, y, x, v)
    @test log!(MM2, v, x, y) === log!(M, v, x, y)
    @test retract!(MM2, y, x, v) === retract!(M, y, x, v)
    @test retract!(MM2, y, x, v, 1) === retract!(M, y, x, v, 1)

    @test project_point!(MM2, y, x) === project_point!(M, y, x)
    @test project_tangent!(MM2, w, x, v) === project_tangent!(M, w, x, v)
    @test vector_transport_to!(MM2, w, x, v, y) == vector_transport_to!(M, w, x, v, y)
    @test zero_tangent_vector!(MM2, v, x) === zero_tangent_vector!(M, v, x)
    @test injectivity_radius(MM2, x) === injectivity_radius(M, x)
    @test injectivity_radius(MM2) === injectivity_radius(M)
    @test is_manifold_point(MM2, x) === is_manifold_point(M, x)
    @test is_tangent_vector(MM2, x, v) === is_tangent_vector(M, x, v)

    a = Manifolds.projected_distribution(M, Distributions.MvNormal(zero(zeros(3)), 1.0))
    b = Manifolds.projected_distribution(MM2, Distributions.MvNormal(zero(zeros(3)), 1.0))
    @test isapprox(Matrix(a.d.Σ), Matrix(b.d.Σ))
    @test isapprox(a.d.μ, b.d.μ)
    @test tangent_orthonormal_basis(M,x,v) == tangent_orthonormal_basis(MM2,x,v)
    @test_throws ErrorException tangent_orthonormal_basis(MM,x,v)
    cov = flat(M, x, FVector(TangentSpace, v))
    cow = flat(M, x, FVector(TangentSpace, w))
    @test cov.data ≈ flat(MM, x, FVector(TangentSpace, v)).data
    cotspace = CotangentBundleFibers(M)
    cotspace2 = CotangentBundleFibers(MM)
    @test cov.data ≈ 2 * v
    @test inner(M, x, v, w) ≈ inner(cotspace, x, cov.data, cow.data)
    @test inner(MM, x, v, w) ≈ inner(cotspace, x, cov.data, cow.data)
    @test inner(MM, x, v, w) ≈ inner(cotspace2, x, cov.data, cow.data)
    @test sharp(M, x, cov).data ≈ v

    xsample = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    w = pweights([0.5, 0.5])
    @test mean(M, xsample, w) ≈ ones(3)
    @test mean(MM2, xsample, w) ≈ ones(3)
    @test mean(MM, xsample, w) ≈ 3 .* ones(3)

    @test median(M, xsample, w) ≈ 2 .* ones(3)
    @test median(MM2, xsample, w) ≈ 2 * ones(3)
    @test median(MM, xsample, w) ≈ 4 .* ones(3)
end
