using FiniteDifferences
using LinearAlgebra: I
using StatsBase: AbstractWeights, pweights
using ManifoldsBase: TraitList
import ManifoldsBase: default_retraction_method
import Manifolds: solve_exp_ode
using Manifolds: InducedBasis, connection, get_chart_index, induced_basis, mean!, median!
using ManifoldDiff: FiniteDifferencesBackend
include("utils.jl")

struct TestEuclidean{N} <: AbstractManifold{ℝ} end
struct TestEuclideanMetric <: AbstractMetric end
struct TestScaledEuclideanMetric <: AbstractMetric end
struct TestRetraction <: AbstractRetractionMethod end
struct TestConnection <: AbstractAffineConnection end

ManifoldsBase.default_retraction_method(::TestEuclidean) = TestRetraction()
function ManifoldsBase.default_retraction_method(
    ::MetricManifold{ℝ,<:TestEuclidean,<:TestEuclideanMetric},
)
    return TestRetraction()
end

Manifolds.manifold_dimension(::TestEuclidean{N}) where {N} = N
function Manifolds.local_metric(
    ::TraitList{<:IsMetricManifold},
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestEuclideanMetric},
    ::Any,
    ::InducedBasis,
)
    return Diagonal(1.0:manifold_dimension(M))
end
function Manifolds.local_metric(
    ::TraitList{IsMetricManifold},
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestEuclideanMetric},
    ::Any,
    ::T,
) where {T<:ManifoldsBase.AbstractOrthogonalBasis}
    return Diagonal(1.0:manifold_dimension(M))
end
function Manifolds.local_metric(
    ::TraitList{IsMetricManifold},
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestScaledEuclideanMetric},
    ::Any,
    ::T,
) where {T<:ManifoldsBase.AbstractOrthogonalBasis}
    return 2 .* Diagonal(1.0:manifold_dimension(M))
end
function Manifolds.get_coordinates_orthogonal(
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestEuclideanMetric},
    ::Any,
    X,
    ::ManifoldsBase.AbstractNumbers,
)
    return 1 ./ [1.0:manifold_dimension(M)...] .* X
end
function Manifolds.get_coordinates_orthogonal!(
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestEuclideanMetric},
    c,
    ::Any,
    X,
    ::ManifoldsBase.AbstractNumbers,
)
    c .= 1 ./ [1.0:manifold_dimension(M)...] .* X
    return c
end
function Manifolds.get_vector_orthonormal!(
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestEuclideanMetric},
    X,
    ::Any,
    c,
    ::ManifoldsBase.AbstractNumbers,
)
    X .= [1.0:manifold_dimension(M)...] .* c
    return X
end
function Manifolds.get_coordinates_orthogonal!(
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestScaledEuclideanMetric},
    c,
    ::Any,
    X,
)
    c .= 1 ./ (2 .* [1.0:manifold_dimension(M)...]) .* X
    return c
end
function Manifolds.get_vector_orthogonal!(
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestScaledEuclideanMetric},
    ::Any,
    c,
    ::ManifoldsBase.AbstractNumbers,
)
    return 2 .* [1.0:manifold_dimension(M)...] .* c
end
function Manifolds.get_vector_orthogonal!(
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestScaledEuclideanMetric},
    X,
    ::Any,
    c,
    ::ManifoldsBase.AbstractNumbers,
)
    X .= 2 .* [1.0:manifold_dimension(M)...] .* c
    return X
end
struct TestSphere{N,T} <: AbstractManifold{ℝ}
    r::T
end

struct TestSphericalMetric <: AbstractMetric end

Manifolds.manifold_dimension(::TestSphere{N}) where {N} = N
function Manifolds.local_metric(
    M::MetricManifold{ℝ,<:TestSphere,<:TestSphericalMetric},
    p,
    ::InducedBasis,
)
    r = base_manifold(M).r
    d = allocate(p)
    d[1] = r^2
    d[2] = d[1] * sin(p[1])^2
    return Diagonal(d)
end
sph_to_cart(θ, ϕ) = [cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)]

struct BaseManifold{N} <: AbstractManifold{ℝ} end
struct BaseManifoldMetric{M} <: AbstractMetric end
struct DefaultBaseManifoldMetric <: AbstractMetric end
struct NotImplementedMetric <: AbstractMetric end

Manifolds.manifold_dimension(::BaseManifold{N}) where {N} = N
Manifolds.inner(::BaseManifold, p, X, Y) = 2 * dot(X, Y)
Manifolds.exp!(::BaseManifold, q, p, X) = q .= p + 2 * X
Manifolds.exp!(::BaseManifold, q, p, X, t::Number) = q .= p + 2 * t * X
Manifolds.log!(::BaseManifold, Y, p, q) = Y .= (q - p) / 2
Manifolds.project!(::BaseManifold, Y, p, X) = Y .= 2 .* X
Manifolds.project!(::BaseManifold, q, p) = (q .= p)
Manifolds.injectivity_radius(::BaseManifold) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::Any) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::AbstractRetractionMethod) = Inf
Manifolds._injectivity_radius(::BaseManifold, ::ExponentialRetraction) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::Any, ::AbstractRetractionMethod) = Inf
Manifolds._injectivity_radius(::BaseManifold, ::Any, ::ExponentialRetraction) = Inf
function Manifolds.local_metric(
    ::TraitList{<:IsMetricManifold},
    ::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    p,
    ::InducedBasis,
) where {N}
    return 2 * one(p * p')
end
function Manifolds.exp!(
    M::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    q,
    p,
    X,
) where {N}
    return exp!(base_manifold(M), q, p, X)
end
function Manifolds.exp!(
    M::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    q,
    p,
    X,
    t::Number,
) where {N}
    return exp!(base_manifold(M), q, p, X, t)
end
function Manifolds.parallel_transport_to!(::BaseManifold, Y, p, X, q)
    return (Y .= X)
end
function Manifolds.get_basis(
    ::BaseManifold{N},
    p,
    B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType},
) where {N}
    return CachedBasis(B, [(Matrix{eltype(p)}(I, N, N)[:, i]) for i in 1:N])
end
function Manifolds.get_coordinates_orthonormal!(
    ::BaseManifold,
    Y,
    p,
    X,
    ::ManifoldsBase.AbstractNumbers,
)
    return Y .= X
end
function Manifolds.get_vector_orthonormal!(
    ::BaseManifold,
    Y,
    p,
    X,
    ::ManifoldsBase.AbstractNumbers,
)
    return Y .= X
end
Manifolds.is_default_metric(::BaseManifold, ::DefaultBaseManifoldMetric) = true
function Manifolds.projected_distribution(M::BaseManifold, d)
    return ProjectedPointDistribution(M, d, project!, rand(d))
end
function Manifolds.projected_distribution(M::BaseManifold, d, p)
    return ProjectedPointDistribution(M, d, project!, p)
end

function Manifolds.flat!(
    ::BaseManifold,
    v::FVector{Manifolds.CotangentSpaceType},
    x,
    w::FVector{Manifolds.TangentSpaceType},
)
    v.data .= 2 .* w.data
    return v
end
function Manifolds.sharp!(
    ::BaseManifold,
    v::FVector{Manifolds.TangentSpaceType},
    x,
    w::FVector{Manifolds.CotangentSpaceType},
)
    v.data .= w.data ./ 2
    return v
end
function solve_exp_ode(
    ::ConnectionManifold{ℝ,TestEuclidean{N},TestConnection},
    p,
    X,
    t::Number;
    kwargs...,
) where {N}
    return X
end
function Manifolds.vector_transport_along!(
    M::BaseManifold,
    Y,
    p,
    X,
    c::AbstractVector,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M),
)
    Y .= c
    return Y
end

# test for https://github.com/JuliaManifolds/Manifolds.jl/issues/539
struct Issue539Metric <: RiemannianMetric end
Manifolds.inner(::MetricManifold{ℝ,<:AbstractManifold{ℝ},Issue539Metric}, p, X, Y) = 3

@testset "Metrics" begin
    # some tests failed due to insufficient accuracy for a particularly bad RNG state
    Random.seed!(42)
    @testset "Metric Basics" begin
        @test repr(MetricManifold(Euclidean(3), EuclideanMetric())) ===
              "MetricManifold(Euclidean(3; field = ℝ), EuclideanMetric())"
        @test repr(IsDefaultMetric(EuclideanMetric())) ===
              "IsDefaultMetric(EuclideanMetric())"
    end
    @testset "Connection Trait" begin
        M = ConnectionManifold(Euclidean(3), LeviCivitaConnection())
        @test is_default_connection(M)
        @test decorated_manifold(M) == Euclidean(3)
        @test is_default_connection(Euclidean(3), LeviCivitaConnection())
        @test !is_default_connection(TestEuclidean{3}(), LeviCivitaConnection())
        c = IsDefaultConnection(LeviCivitaConnection())
        @test ManifoldsBase.parent_trait(c) == Manifolds.IsConnectionManifold()
    end

    @testset "solve_exp_ode error message" begin
        E = TestEuclidean{3}()
        g = TestEuclideanMetric()
        M = MetricManifold(E, g)
        default_retraction_method(::TestEuclidean) = TestRetraction()
        p = [1.0, 2.0, 3.0]
        X = [2.0, 3.0, 4.0]
        q = similar(X)
        @test_throws MethodError exp(M, p, X)
        @test_throws MethodError exp(M, p, X, 1.0)
        @test_throws MethodError exp!(M, q, p, X)
        @test_throws MethodError exp!(M, q, p, X, 1.0)

        N = ConnectionManifold(E, LeviCivitaConnection())
        @test_throws MethodError exp(N, p, X)
        @test_throws MethodError exp(N, p, X, 1.0)
        @test_throws MethodError exp!(N, q, p, X)
        @test_throws MethodError exp!(N, q, p, X, 1.0)

        using OrdinaryDiffEq
        @test is_point(M, exp(M, p, X))
        @test is_point(M, exp(M, p, X, 1.0))

        # a small trick to check that retract_exp_ode! returns the right value on ConnectionManifolds
        N2 = ConnectionManifold(E, TestConnection())
        @test exp(N2, p, X) == X
    end
    @testset "Local Metric Error message" begin
        M = MetricManifold(BaseManifold{2}(), NotImplementedMetric())
        A = Manifolds.get_default_atlas(M)
        p = [3, 4]
        i = get_chart_index(M, A, p)

        B = induced_basis(M, A, i, TangentSpace)
        @test_throws MethodError local_metric(M, p, B)
    end
    @testset "scaled Euclidean metric" begin
        n = 3
        E = TestEuclidean{n}()
        g = TestEuclideanMetric()
        M = MetricManifold(E, g)
        A = Manifolds.get_default_atlas(M)
        @test repr(M) == "MetricManifold(TestEuclidean{3}(), TestEuclideanMetric())"

        @test TestEuclideanMetric()(E) === M
        @test TestEuclideanMetric(E) === M
        @test connection(M) === Manifolds.LeviCivitaConnection()

        G = Diagonal(1.0:n)
        invG = inv(G)
        @test manifold_dimension(M) == n
        @test base_manifold(M) === E
        @test metric(M) === g

        i_zeros = get_chart_index(M, A, zeros(3))
        B_i_zeros = induced_basis(M, A, i_zeros, TangentSpace)
        @test_throws MethodError local_metric_jacobian(E, zeros(3), B_i_zeros)
        @test_throws MethodError christoffel_symbols_second_jacobian(E, zeros(3), B_i_zeros)

        for vtype in (Vector, MVector{n})
            p, X, Y = vtype(randn(n)), vtype(randn(n)), vtype(randn(n))

            chart_p = get_chart_index(M, A, p)
            B_chart_p = induced_basis(M, A, chart_p, TangentSpace)

            @test check_point(M, p) == check_point(E, p)
            @test check_vector(M, p, X) == check_vector(E, p, X)

            @test local_metric(M, p, B_chart_p) ≈ G
            @test inverse_local_metric(M, p, B_chart_p) ≈ invG
            @test det_local_metric(M, p, B_chart_p) ≈ *(1.0:n...)
            @test log_local_metric_density(M, p, B_chart_p) ≈ sum(log.(1.0:n)) / 2

            fX = ManifoldsBase.TFVector(X, B_chart_p)
            fY = ManifoldsBase.TFVector(Y, B_chart_p)
            @test inner(M, p, fX, fY) ≈ dot(X, G * Y) atol = 1e-6
            @test norm(M, p, fX) ≈ sqrt(dot(X, G * X)) atol = 1e-6

            @test christoffel_symbols_first(M, p, B_chart_p) ≈ zeros(n, n, n) atol = 1e-6
            @test christoffel_symbols_second(M, p, B_chart_p) ≈ zeros(n, n, n) atol = 1e-6
            @test riemann_tensor(M, p, B_chart_p) ≈ zeros(n, n, n, n) atol = 1e-6
            @test ricci_tensor(M, p, B_chart_p) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, p, B_chart_p) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, p, B_chart_p) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, p, B_chart_p) ≈ zeros(n, n) atol = 1e-6

            fdm = FiniteDifferencesBackend(forward_fdm(2, 1))
            @test christoffel_symbols_first(M, p, B_chart_p; backend=fdm) ≈ zeros(n, n, n) atol =
                1e-6
            @test christoffel_symbols_second(M, p, B_chart_p; backend=fdm) ≈ zeros(n, n, n) atol =
                1e-6
            @test riemann_tensor(M, p, B_chart_p; backend=fdm) ≈ zeros(n, n, n, n) atol =
                1e-6
            @test ricci_tensor(M, p, B_chart_p; backend=fdm) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, p, B_chart_p; backend=fdm) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, p, B_chart_p; backend=fdm) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, p, B_chart_p; backend=fdm) ≈ zeros(n, n) atol = 1e-6

            fd_diff = FiniteDifferencesBackend()
            @test christoffel_symbols_first(M, p, B_chart_p; backend=fd_diff) ≈
                  zeros(n, n, n) atol = 1e-6
            @test christoffel_symbols_second(M, p, B_chart_p; backend=fd_diff) ≈
                  zeros(n, n, n) atol = 1e-6
            @test riemann_tensor(M, p, B_chart_p; backend=fd_diff) ≈ zeros(n, n, n, n) atol =
                1e-6
            @test ricci_tensor(M, p, B_chart_p; backend=fd_diff) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, p, B_chart_p; backend=fd_diff) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, p, B_chart_p; backend=fd_diff) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, p, B_chart_p; backend=fd_diff) ≈ zeros(n, n) atol =
                1e-6
        end
    end

    @testset "default_* functions" begin
        E = Euclidean(3)
        EM = MetricManifold(E, EuclideanMetric())
        @test default_retraction_method(EM) === default_retraction_method(E)
        @test default_inverse_retraction_method(EM) === default_inverse_retraction_method(E)
        @test default_vector_transport_method(EM) === default_vector_transport_method(E)
        EC = ConnectionManifold(E, TestConnection())
        @test default_retraction_method(EC) === default_retraction_method(E)
        @test default_inverse_retraction_method(EC) === default_inverse_retraction_method(E)
        @test default_vector_transport_method(EC) === default_vector_transport_method(E)
    end

    @testset "scaled Sphere metric" begin
        n = 2
        r = 10 * rand()
        θ, ϕ = π * rand(), 2π * rand()
        Sr = TestSphere{n,Float64}(r)
        S = Manifolds.Sphere(n)
        g = TestSphericalMetric()
        M = MetricManifold(Sr, g)
        A = Manifolds.get_default_atlas(M)

        @test manifold_dimension(M) == n
        @test base_manifold(M) === Sr
        @test metric(M) === g

        for vtype in (Vector, MVector{n})
            p = vtype([θ, ϕ])
            chart_p = get_chart_index(M, A, p)
            B_p = induced_basis(M, A, chart_p, TangentSpace)
            G = Diagonal(vtype([1, sin(θ)^2])) .* r^2
            invG = Diagonal(vtype([1, 1 / sin(θ)^2])) ./ r^2
            X, Y = normalize(randn(n)), normalize(randn(n))

            @test local_metric(M, p, B_p) ≈ G atol = 1e-6
            @test inverse_local_metric(M, p, B_p) ≈ invG atol = 1e-6
            @test det_local_metric(M, p, B_p) ≈ r^4 * sin(θ)^2 atol = 1e-6
            @test log_local_metric_density(M, p, B_p) ≈ 2 * log(r) + log(sin(θ)) atol = 1e-6
            fX = ManifoldsBase.TFVector(X, B_p)
            fY = ManifoldsBase.TFVector(Y, B_p)
            @test inner(M, p, fX, fY) ≈ dot(X, G * Y) atol = 1e-6
            @test norm(M, p, fX) ≈ sqrt(dot(X, G * X)) atol = 1e-6

            pcart = sph_to_cart(θ, ϕ)
            Xcart = [
                cos(ϕ)*cos(θ) -sin(ϕ)*sin(θ)
                sin(ϕ)*cos(θ) cos(ϕ)*sin(θ)
                -sin(θ) 0
            ] * X

            Γ₁ = christoffel_symbols_first(M, p, B_p)
            for i in 1:n, j in 1:n, k in 1:n
                if (i, j, k) == (1, 2, 2) || (i, j, k) == (2, 1, 2)
                    @test Γ₁[i, j, k] ≈ r^2 * cos(θ) * sin(θ) atol = 1e-6
                elseif (i, j, k) == (2, 2, 1)
                    @test Γ₁[i, j, k] ≈ -r^2 * cos(θ) * sin(θ) atol = 1e-6
                else
                    @test Γ₁[i, j, k] ≈ 0 atol = 1e-6
                end
            end

            Γ₂ = christoffel_symbols_second(M, p, B_p)
            for l in 1:n, i in 1:n, j in 1:n
                if (l, i, j) == (1, 2, 2)
                    @test Γ₂[l, i, j] ≈ -cos(θ) * sin(θ) atol = 1e-6
                elseif (l, i, j) == (2, 1, 2) || (l, i, j) == (2, 2, 1)
                    @test Γ₂[l, i, j] ≈ cot(θ) atol = 1e-6
                else
                    @test Γ₂[l, i, j] ≈ 0 atol = 1e-6
                end
            end

            R = riemann_tensor(M, p, B_p)
            for l in 1:n, i in 1:n, j in 1:n, k in 1:n
                if (l, i, j, k) == (2, 1, 1, 2)
                    @test R[l, i, j, k] ≈ -1 atol = 2e-6
                elseif (l, i, j, k) == (2, 1, 2, 1)
                    @test R[l, i, j, k] ≈ 1 atol = 2e-6
                elseif (l, i, j, k) == (1, 2, 1, 2)
                    @test R[l, i, j, k] ≈ sin(θ)^2 atol = 1e-6
                elseif (l, i, j, k) == (1, 2, 2, 1)
                    @test R[l, i, j, k] ≈ -sin(θ)^2 atol = 1e-6
                else
                    @test R[l, i, j, k] ≈ 0 atol = 1e-6
                end
            end

            @test ricci_tensor(M, p, B_p) ≈ G ./ r^2 atol = 2e-6
            @test ricci_curvature(M, p, B_p) ≈ 2 / r^2 atol = 2e-6
            @test gaussian_curvature(M, p, B_p) ≈ 1 / r^2 atol = 2e-6
            @test einstein_tensor(M, p, B_p) ≈
                  ricci_tensor(M, p, B_p) - gaussian_curvature(M, p, B_p) .* G atol = 1e-6
        end
    end

    @testset "Metric decorator" begin
        M = BaseManifold{3}()
        g = BaseManifoldMetric{3}()
        MM = MetricManifold(M, g)

        @test DefaultBaseManifoldMetric(BaseManifold{3}()) ===
              MetricManifold(BaseManifold{3}(), DefaultBaseManifoldMetric())
        MT = DefaultBaseManifoldMetric()
        @test MT(BaseManifold{3}()) ===
              MetricManifold(BaseManifold{3}(), DefaultBaseManifoldMetric())

        g2 = DefaultBaseManifoldMetric()
        MM2 = MetricManifold(M, g2)
        A = Manifolds.get_default_atlas(M)

        @test is_default_metric(MM) == is_default_metric(base_manifold(MM), metric(MM))
        @test is_default_metric(MM2) == is_default_metric(base_manifold(MM2), metric(MM2))
        @test is_default_metric(MM2)

        @test convert(typeof(MM2), M) == MM2
        @test_throws ErrorException convert(typeof(MM), M)
        p = [0.1, 0.2, 0.4]
        X = [0.5, 0.7, 0.11]
        Y = [0.13, 0.17, 0.19]
        q = allocate(p)

        p2 = allocate(p)
        copyto!(MM, p2, p)
        p3 = allocate(p)
        copyto!(M, p3, p)
        @test p2 == p3
        X = zero_vector(MM, p)
        Y = allocate(X)
        copyto!(MM, Y, p, X)
        Y2 = allocate(X)
        copyto!(M, Y2, p, X)
        @test Y == Y2

        X = [0.5, 0.7, 0.11]

        chart_p = get_chart_index(M, A, p)
        B_p = induced_basis(M, A, chart_p, TangentSpace)
        fX = ManifoldsBase.TFVector(X, B_p)
        fY = ManifoldsBase.TFVector(Y, B_p)

        @test inner(M, p, X, Y) == 2 * dot(X, Y)
        @test inner(MM, p, fX, fY) === inner(M, p, X, Y)
        @test norm(MM, p, fX) === norm(M, p, X)
        @test exp(M, p, X) == p + 2 * X
        @test exp(M, p, X, 0.5) == p + X
        @test exp(MM2, p, X) == exp(M, p, X)
        @test exp(MM2, p, X, 0.5) == exp(M, p, X, 0.5)
        @test exp!(MM, q, p, X) === exp!(M, q, p, X)
        @test exp!(MM, q, p, X, 0.5) === exp!(M, q, p, X, 0.5)
        @test retract!(MM, q, p, X) === retract!(M, q, p, X)
        @test retract!(MM, q, p, X, 1) === retract!(M, q, p, X, 1)
        @test project!(MM, Y, p, X) === project!(M, Y, p, X)
        @test project!(MM, q, p) === project!(M, q, p)
        # without a definition for the metric from the embedding, no projection possible
        @test_throws MethodError log!(MM, Y, p, q) === project!(M, Y, p, q)
        @test_throws MethodError vector_transport_to!(MM, Y, p, X, q) ===
                                 vector_transport_to!(M, Y, p, X, q)
        # without DiffEq, these error
        @test_throws MethodError exp(MM, p, X, 1:3)
        # these always fall back anyways.
        @test zero_vector!(MM, X, p) === zero_vector!(M, X, p)

        @test injectivity_radius(MM, p) === injectivity_radius(M, p)
        @test injectivity_radius(MM) === injectivity_radius(M)
        @test injectivity_radius(MM, ProjectionRetraction()) ===
              injectivity_radius(M, ProjectionRetraction())
        @test injectivity_radius(MM, ExponentialRetraction()) ===
              injectivity_radius(M, ExponentialRetraction())
        @test injectivity_radius(MM) === injectivity_radius(M)

        @test is_point(MM, p) === is_point(M, p)
        @test is_vector(MM, p, X) === is_vector(M, p, X)

        A = Manifolds.get_default_atlas(MM2)
        chart_p = get_chart_index(MM2, A, p)
        B_p = induced_basis(MM2, A, chart_p, TangentSpace)
        @test_throws MethodError local_metric(MM2, p, B_p)
        @test_throws MethodError local_metric_jacobian(MM2, p, B_p)
        @test_throws MethodError christoffel_symbols_second_jacobian(MM2, p, B_p)
        # MM falls back to nondefault error
        @test_throws MethodError projected_distribution(MM, 1, p)
        @test_throws MethodError projected_distribution(MM, 1)

        @test inner(MM2, p, X, Y) === inner(M, p, X, Y)
        @test norm(MM2, p, X) === norm(M, p, X)
        @test distance(MM2, p, q) === distance(M, p, q)
        @test exp!(MM2, q, p, X) === exp!(M, q, p, X)
        @test exp(MM2, p, X) == exp(M, p, X)
        @test log!(MM2, X, p, q) === log!(M, X, p, q)
        @test log(MM2, p, q) == log(M, p, q)
        @test retract!(MM2, q, p, X) === retract!(M, q, p, X)
        @test retract!(MM2, q, p, X, 1) === retract!(M, q, p, X, 1)

        @test project!(MM2, q, p) === project!(M, q, p)
        @test project!(MM2, Y, p, X) === project!(M, Y, p, X)
        @test parallel_transport_to(MM2, p, X, q) == parallel_transport_to(M, q, X, p)
        @test parallel_transport_to!(MM2, Y, p, X, q) ==
              parallel_transport_to!(M, Y, q, X, p)
        @test project!(MM2, Y, p, X) === project!(M, Y, p, X)
        @test vector_transport_to!(MM2, Y, p, X, q) == vector_transport_to!(M, Y, p, X, q)
        c = 2 * ones(3)
        m = ParallelTransport()
        @test vector_transport_along(MM2, p, X, c, m) ==
              vector_transport_along(M, p, X, c, m)
        @test vector_transport_along!(MM2, Y, p, X, c, m) ==
              vector_transport_along!(M, Y, p, X, c, m)
        @test zero_vector!(MM2, X, p) === zero_vector!(M, X, p)
        @test injectivity_radius(MM2, p) === injectivity_radius(M, p)
        @test injectivity_radius(MM2) === injectivity_radius(M)
        @test injectivity_radius(MM2, p, ExponentialRetraction()) ===
              injectivity_radius(M, p, ExponentialRetraction())
        @test injectivity_radius(MM2, ExponentialRetraction()) ===
              injectivity_radius(M, ExponentialRetraction())
        @test injectivity_radius(MM2, p, ProjectionRetraction()) ===
              injectivity_radius(M, p, ProjectionRetraction())
        @test injectivity_radius(MM2, ProjectionRetraction()) ===
              injectivity_radius(M, ProjectionRetraction())
        @test is_point(MM2, p) === is_point(M, p)
        @test is_vector(MM2, p, X) === is_vector(M, p, X)

        a = Manifolds.projected_distribution(M, Distributions.MvNormal(zero(zeros(3)), 1.0))
        b = Manifolds.projected_distribution(
            MM2,
            Distributions.MvNormal(zero(zeros(3)), 1.0),
        )
        @test isapprox(Matrix(a.distribution.Σ), Matrix(b.distribution.Σ))
        @test isapprox(a.distribution.μ, b.distribution.μ)
        @test get_basis(M, p, DefaultOrthonormalBasis()).data ==
              get_basis(MM2, p, DefaultOrthonormalBasis()).data
        @test_throws MethodError get_basis(MM, p, DefaultOrthonormalBasis())

        fX = ManifoldsBase.TFVector(X, B_p)
        fY = ManifoldsBase.TFVector(Y, B_p)
        coX = flat(M, p, X)
        coY = flat(M, p, Y)
        cofX = flat(M, p, fX)
        cofY = flat(M, p, fY)
        @test coX(X) ≈ norm(M, p, X)^2
        @test coY(X) ≈ inner(M, p, X, Y)
        cotspace = CotangentBundleFibers(M)
        cotspace2 = CotangentBundleFibers(MM)
        @test coX.X ≈ X
        @test inner(M, p, X, Y) ≈ inner(cotspace, p, coX, coY)
        @test inner(MM, p, fX, fY) ≈ inner(cotspace, p, coX, coY)

        @test inner(MM, p, fX, fY) ≈ inner(cotspace2, p, cofX, cofY)
        @test sharp(M, p, coX) ≈ X

        coMMfX = flat(MM, p, fX)
        coMMfY = flat(MM, p, fY)
        @test inner(MM, p, fX, fY) ≈ inner(cotspace2, p, coMMfX, coMMfY)
        @test isapprox(sharp(MM, p, coMMfX).data, fX.data)

        @testset "Mutating flat/sharp" begin
            cofX2 = allocate(cofX)
            flat!(M, cofX2, p, fX)
            @test isapprox(cofX2.data, cofX.data)

            fX2 = allocate(fX)
            sharp!(M, fX2, p, cofX2)
            @test isapprox(fX2.data, fX.data)

            cofX2 = allocate(cofX)
            flat!(MM, cofX2, p, fX)
            @test isapprox(cofX2.data, cofX.data)

            fX2 = allocate(fX)
            sharp!(MM, fX2, p, cofX2)
            @test isapprox(fX2.data, fX.data)

            cofX3a = flat(MM2, p, fX)
            cofX3b = allocate(cofX3a)
            flat!(MM2, cofX3b, p, fX)
            @test isapprox(cofX3a.data, cofX3b.data)

            fX3a = sharp(MM2, p, cofX)
            fX3b = allocate(fX3a)
            sharp!(MM2, fX3b, p, cofX)
            @test isapprox(fX3a.data, fX3b.data)
        end
    end

    @testset "change metric and representer" begin
        M = MetricManifold(TestEuclidean{2}(), TestEuclideanMetric())
        G = TestScaledEuclideanMetric()
        M2 = TestScaledEuclideanMetric(M)
        @test M2.manifold === M.manifold
        @test M2.metric == G
        p = ones(2)
        X = 2 * ones(2)
        @test change_metric(M, TestEuclideanMetric(), p, X) == X
        Y = change_metric(M, G, p, X)
        @test Y ≈ sqrt(2) .* X #scaled metric has a factor 2, removing introduces this factor
        @test change_representer(M, TestEuclideanMetric(), p, X) == X
        Y2 = change_representer(M, G, p, X)
        @test Y2 ≈ 2 .* X #scaled metric has a factor 2, removing introduces this factor
    end

    @testset "issue #539" begin
        M = Sphere(2)
        p = [0.49567358314486515, 0.3740229181343087, -0.7838460025302334]
        X = [-1.1552859627097727, 0.40665559717366767, -0.5365163797547751]
        MM = MetricManifold(M, Issue539Metric())
        @test norm(MM, p, X)^2 ≈ 3
        @test Manifolds._drop_embedding_type(
            ManifoldsBase.merge_traits(IsEmbeddedSubmanifold()),
        ) === ManifoldsBase.EmptyTrait()
        @test get_embedding(MM) === get_embedding(M)
    end
end
