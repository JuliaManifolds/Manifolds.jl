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
    _gradient!

using FiniteDifferences
using LinearAlgebra: Diagonal, dot

@testset "Differentiation backend" begin
    fd51 = Manifolds.FiniteDifferencesBackend()
    @testset "diff_backend" begin
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend
        @test length(diff_backends()) == 3
        @test diff_backends()[1] isa Manifolds.FiniteDifferencesBackend

        @test length(fd51.method.grid) == 5
        # check method order
        @test typeof(fd51.method).parameters[2] == 1
        fd71 = Manifolds.FiniteDifferencesBackend(central_fdm(7, 1))
        @test diff_backend!(fd71) == fd71
        @test diff_backend() == fd71
    end

    using ForwardDiff

    fwd_diff = Manifolds.ForwardDiffBackend()
    @testset "ForwardDiff" begin
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend
        @test length(diff_backends()) == 3
        @test diff_backends()[1] isa Manifolds.FiniteDifferencesBackend
        @test diff_backends()[2] == fwd_diff

        @test diff_backend!(fwd_diff) == fwd_diff
        @test diff_backend() == fwd_diff
        @test diff_backend!(fd51) isa Manifolds.FiniteDifferencesBackend
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend

        diff_backend!(fwd_diff)
        @test diff_backend() == fwd_diff
        diff_backend!(fd51)
    end

    using FiniteDiff

    finite_diff = Manifolds.FiniteDiffBackend()
    @testset "FiniteDiff" begin
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend
        @test length(diff_backends()) == 4
        @test diff_backends()[4] == finite_diff

        @test diff_backend!(finite_diff) == finite_diff
        @test diff_backend() == finite_diff
        @test diff_backend!(fd51) isa Manifolds.FiniteDifferencesBackend
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend

        diff_backend!(finite_diff)
        @test diff_backend() == finite_diff
        diff_backend!(fd51)
    end

    using ReverseDiff

    reverse_diff = Manifolds.ReverseDiffBackend()
    @testset "ReverseDiff" begin
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend
        @test length(diff_backends()) == 4
        @test diff_backends()[3] == reverse_diff

        @test diff_backend!(reverse_diff) == reverse_diff
        @test diff_backend() == reverse_diff
        @test diff_backend!(fd51) isa Manifolds.FiniteDifferencesBackend
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend

        diff_backend!(reverse_diff)
        @test diff_backend() == reverse_diff
        diff_backend!(fd51)
    end

    using Zygote: Zygote

    zygote_diff = Manifolds.ZygoteDiffBackend()
    @testset "Zygote" begin
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend
        @test length(diff_backends()) == 5
        @test diff_backends()[5] == zygote_diff

        @test diff_backend!(zygote_diff) == zygote_diff
        @test diff_backend() == zygote_diff
        @test diff_backend!(fd51) isa Manifolds.FiniteDifferencesBackend
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend

        diff_backend!(zygote_diff)
        @test diff_backend() == zygote_diff
        diff_backend!(fd51)
    end

    @testset "gradient" begin
        diff_backend!(fd51)
        r2 = Euclidean(2)

        c1(t) = [sin(t), cos(t)]
        f1(x) = x[1] + x[2]^2
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
            diff_backend!(backend)
            @test _derivative(c1, 0.0) ≈ [1.0, 0.0]
            X = [-1.0, -1.0]
            @test _derivative!(c1, X, 0.0) === X
            @test isapprox(X, [1.0, 0.0])
        end
        @testset for backend in [fd51, fwd_diff, finite_diff, reverse_diff, zygote_diff]
            diff_backend!(backend)
            X = [-1.0, -1.0]
            @test _gradient(f1, [1.0, -1.0]) ≈ [1.0, -2.0]
            @test _gradient!(f1, X, [1.0, -1.0]) === X
            @test X ≈ [1.0, -2.0]
        end
        diff_backend!(Manifolds.NoneDiffBackend())
        @testset for backend in [fd51, Manifolds.ForwardDiffBackend()]
            @test _derivative(c1, 0.0, backend) ≈ [1.0, 0.0]
            @test _gradient(f1, [1.0, -1.0], backend) ≈ [1.0, -2.0]
        end

        diff_backend!(fd51)
    end
end

rb_onb_default = RiemannianONBDiffBackend(
    diff_backend(),
    Manifolds.ExponentialRetraction(),
    Manifolds.LogarithmicInverseRetraction(),
    DefaultOrthonormalBasis(),
)

rb_onb_fd51 = RiemannianONBDiffBackend(
    Manifolds.FiniteDifferencesBackend(),
    Manifolds.ExponentialRetraction(),
    Manifolds.LogarithmicInverseRetraction(),
    DefaultOrthonormalBasis(),
)

rb_onb_fwd_diff = RiemannianONBDiffBackend(
    Manifolds.ForwardDiffBackend(),
    Manifolds.ExponentialRetraction(),
    Manifolds.LogarithmicInverseRetraction(),
    DefaultOrthonormalBasis(),
)

rb_onb_finite_diff = RiemannianONBDiffBackend(
    Manifolds.FiniteDiffBackend(),
    Manifolds.ExponentialRetraction(),
    Manifolds.LogarithmicInverseRetraction(),
    DefaultOrthonormalBasis(),
)

rb_onb_default2 = RiemannianONBDiffBackend(
    diff_backend(),
    Manifolds.ExponentialRetraction(),
    Manifolds.LogarithmicInverseRetraction(),
    CachedBasis(
        DefaultOrthonormalBasis(),
        [[0.0, -1.0, 0.0], [sqrt(2) / 2, 0.0, -sqrt(2) / 2]],
    ),
)

rb_proj = Manifolds.RiemannianProjectionGradientBackend(diff_backend())

@testset "rdiff_ functions" begin
    @test Manifolds.rdifferential_backend() ===
          Manifolds._current_rdifferential_backend.backend
    @test Manifolds.rgradient_backend() === Manifolds._current_rgradient_backend.backend

    tmp_diff = Manifolds.rdifferential_backend()
    Manifolds.rdifferential_backend!(rb_onb_finite_diff)
    @test Manifolds.rdifferential_backend() === rb_onb_finite_diff
    Manifolds.rdifferential_backend!(tmp_diff)

    tmp_grad = Manifolds.rgradient_backend()
    Manifolds.rgradient_backend!(rb_onb_finite_diff)
    @test Manifolds.rgradient_backend() === rb_onb_finite_diff
    Manifolds.rgradient_backend!(tmp_grad)
end

@testset "Riemannian differentials" begin
    s2 = Sphere(2)
    p = [0.0, 0.0, 1.0]
    q = [1.0, 0.0, 0.0]
    c1(t) = geodesic(s2, q, p, t)

    Xval = [-sqrt(2) / 2, 0.0, sqrt(2) / 2]
    @test isapprox(s2, c1(π / 4), differential(s2, c1, π / 4), Xval)
    X = similar(p)
    differential!(s2, c1, X, π / 4)
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
    @test isapprox(s2, q, gradient(s2, f1, q), [0.5, 0.0, -0.5])
    for backend in [rb_onb_default, rb_proj]
        @test isapprox(s2, q, gradient(s2, f1, q, backend), [0.5, 0.0, -0.5])
    end
    X = similar(q)
    gradient!(s2, f1, X, q)
    @test isapprox(s2, q, X, [0.5, 0.0, -0.5])
    for backend in [rb_onb_default, rb_proj]
        gradient!(s2, f1, X, q, backend)
        @test isapprox(s2, q, X, [0.5, 0.0, -0.5])
    end
end
