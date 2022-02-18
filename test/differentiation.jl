using Test
using Manifolds
using Manifolds:
    _derivative,
    _derivative!,
    differential,
    differential!,
    gradient,
    gradient!,
    _gradient,
    _gradient!,
    _jacobian,
    _jacobian!

import Manifolds: gradient

struct TestRiemannianBackend <: AbstractRiemannianDiffBackend end
function Manifolds.gradient(::AbstractManifold, f, p, ::TestRiemannianBackend)
    return collect(1.0:length(p))
end

using FiniteDifferences, FiniteDiff
using LinearAlgebra: Diagonal, dot

@testset "Differentiation backend" begin
    fd51 = Manifolds.FiniteDifferencesBackend()
    @testset "default_differential_backend" begin
        #ForwardDiff is loaded first in utils.
        @test default_differential_backend() === Manifolds.ForwardDiffBackend()

        @test length(fd51.method.grid) == 5
        # check method order
        @test typeof(fd51.method).parameters[2] == 1
        fd71 = Manifolds.FiniteDifferencesBackend(central_fdm(7, 1))
        @test set_default_differential_backend!(fd71) == fd71
        @test default_differential_backend() == fd71
    end

    using ForwardDiff

    fwd_diff = Manifolds.ForwardDiffBackend()
    @testset "ForwardDiff" begin
        @test default_differential_backend() isa Manifolds.FiniteDifferencesBackend

        @test set_default_differential_backend!(fwd_diff) == fwd_diff
        @test default_differential_backend() == fwd_diff
        @test set_default_differential_backend!(fd51) isa Manifolds.FiniteDifferencesBackend
        @test default_differential_backend() isa Manifolds.FiniteDifferencesBackend

        set_default_differential_backend!(fwd_diff)
        @test default_differential_backend() == fwd_diff
        set_default_differential_backend!(fd51)
    end

    using FiniteDiff

    finite_diff = Manifolds.FiniteDiffBackend()
    @testset "FiniteDiff" begin
        @test default_differential_backend() isa Manifolds.FiniteDifferencesBackend

        @test set_default_differential_backend!(finite_diff) == finite_diff
        @test default_differential_backend() == finite_diff
        @test set_default_differential_backend!(fd51) isa Manifolds.FiniteDifferencesBackend
        @test default_differential_backend() isa Manifolds.FiniteDifferencesBackend

        set_default_differential_backend!(finite_diff)
        @test default_differential_backend() == finite_diff
        set_default_differential_backend!(fd51)
    end

    using ReverseDiff

    reverse_diff = Manifolds.ReverseDiffBackend()
    @testset "ReverseDiff" begin
        @test default_differential_backend() isa Manifolds.FiniteDifferencesBackend

        @test set_default_differential_backend!(reverse_diff) == reverse_diff
        @test default_differential_backend() == reverse_diff
        @test set_default_differential_backend!(fd51) isa Manifolds.FiniteDifferencesBackend
        @test default_differential_backend() isa Manifolds.FiniteDifferencesBackend

        set_default_differential_backend!(reverse_diff)
        @test default_differential_backend() == reverse_diff
        set_default_differential_backend!(fd51)
    end

    using Zygote
    zygote_diff = Manifolds.ZygoteDiffBackend()

    @testset "gradient" begin
        set_default_differential_backend!(fd51)
        r2 = Euclidean(2)

        c1(t) = [sin(t), cos(t)]
        f1(x) = x[1] + x[2]^2
        function f1!(y, x)
            y .= x[1] + x[2]^2
            return y
        end
        f2(x) = 3 * x[1] * x[2] + x[2]^3

        @testset "Inference" begin
            X = [-1.0, -1.0]
            @test (@inferred _derivative(c1, 0.0, Manifolds.ForwardDiffBackend())) ≈
                  [1.0, 0.0]
            @test (@inferred _derivative!(c1, X, 0.0, Manifolds.ForwardDiffBackend())) === X
            @test X ≈ [1.0, 0.0]

            @test (@inferred _derivative(c1, 0.0, finite_diff)) ≈ [1.0, 0.0]
            @test (@inferred _gradient(f1, [1.0, -1.0], finite_diff)) ≈ [1.0, -2.0]
        end

        @testset for backend in [fd51, fwd_diff, finite_diff]
            set_default_differential_backend!(backend)
            @test _derivative(c1, 0.0) ≈ [1.0, 0.0]
            X = [-1.0, -1.0]
            @test _derivative!(c1, X, 0.0) === X
            @test isapprox(X, [1.0, 0.0])
        end
        @testset for backend in [fd51, fwd_diff, finite_diff, reverse_diff, zygote_diff]
            set_default_differential_backend!(backend)
            X = [-1.0, -1.0]
            @test _gradient(f1, [1.0, -1.0]) ≈ [1.0, -2.0]
            @test _gradient!(f1, X, [1.0, -1.0]) === X
            @test X ≈ [1.0, -2.0]
        end
        @testset for backend in [finite_diff]
            set_default_differential_backend!(backend)
            X = [-0.0 -0.0]
            @test _jacobian(f1, [1.0, -1.0]) ≈ [1.0 -2.0]
            # The following seems not to worf for :central, but it does for forward
            fdf = Manifolds.FiniteDiffBackend(Val(:forward))
            @test _jacobian!(f1!, X, [1.0, -1.0], fdf) === X
            @test X ≈ [1.0 -2.0]
        end
        set_default_differential_backend!(Manifolds.NoneDiffBackend())
        @testset for backend in [fd51, Manifolds.ForwardDiffBackend()]
            @test _derivative(c1, 0.0, backend) ≈ [1.0, 0.0]
            @test _gradient(f1, [1.0, -1.0], backend) ≈ [1.0, -2.0]
        end

        set_default_differential_backend!(fd51)
    end
end

rb_onb_default = TangentDiffBackend(
    default_differential_backend(),
    Manifolds.ExponentialRetraction(),
    Manifolds.LogarithmicInverseRetraction(),
    DefaultOrthonormalBasis(),
)

rb_onb_fd51 = TangentDiffBackend(Manifolds.FiniteDifferencesBackend())

rb_onb_fwd_diff = TangentDiffBackend(Manifolds.ForwardDiffBackend())

rb_onb_finite_diff = TangentDiffBackend(Manifolds.FiniteDiffBackend())

rb_onb_default2 = TangentDiffBackend(
    default_differential_backend();
    basis=CachedBasis(
        DefaultOrthonormalBasis(),
        [[0.0, -1.0, 0.0], [sqrt(2) / 2, 0.0, -sqrt(2) / 2]],
    ),
)

rb_proj = Manifolds.RiemannianProjectionBackend(default_differential_backend())

@testset "Riemannian differentials" begin
    s2 = Sphere(2)
    p = [0.0, 0.0, 1.0]
    q = [1.0, 0.0, 0.0]
    c1(t) = geodesic(s2, q, p, t)

    Xval = [-sqrt(2) / 2, 0.0, sqrt(2) / 2]
    @test isapprox(s2, c1(π / 4), differential(s2, c1, π / 4, rb_onb_default), Xval)
    X = similar(p)
    differential!(s2, c1, X, π / 4, rb_onb_default)
    @test isapprox(s2, c1(π / 4), X, Xval)

    @testset for backend in [rb_onb_fd51, rb_onb_fwd_diff, rb_onb_finite_diff]
        @test isapprox(s2, c1(π / 4), differential(s2, c1, π / 4, backend), Xval)
        X = similar(p)
        differential!(s2, c1, X, π / 4, backend)
        @test isapprox(s2, c1(π / 4), X, Xval)
    end
end

@testset "Riemannian gradients" begin
    s2 = Sphere(2)
    f1(p) = p[1]

    q = [sqrt(2) / 2, 0, sqrt(2) / 2]
    X = similar(q)
    for backend in [rb_onb_default, rb_proj]
        @test isapprox(s2, q, gradient(s2, f1, q, backend), [0.5, 0.0, -0.5])
        @test gradient!(s2, f1, X, q, backend) === X
        @test isapprox(s2, q, X, [0.5, 0.0, -0.5])
    end
    X = similar(q)
    for backend in [rb_onb_default, rb_proj]
        gradient!(s2, f1, X, q, backend)
        @test isapprox(s2, q, X, [0.5, 0.0, -0.5])
    end

    # Test the gradient fallback
    @test gradient(s2, f1, q, TestRiemannianBackend()) == [1.0, 2.0, 3.0]
    X = similar(q)
    @test gradient!(s2, f1, X, q, TestRiemannianBackend()) === X
    @test X == [1.0, 2.0, 3.0]
end

@testset "Default Errors for the ODEExponentialRetraction" begin
    @test_throws DomainError ODEExponentialRetraction(
        ProjectionRetraction(),
        CachedBasis(DefaultOrthogonalBasis(), []), # not yet supported
    )
    @test_throws DomainError ODEExponentialRetraction(
        ExponentialRetraction(), # not possible
        DefaultOrthogonalBasis(),
    )
    @test_throws DomainError ODEExponentialRetraction(
        ExponentialRetraction(), # not possible
        CachedBasis(DefaultOrthogonalBasis(), []), # combination not possible
    )
end
