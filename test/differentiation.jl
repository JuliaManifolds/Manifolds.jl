using Test
using Manifolds
using Manifolds:
    _derivative,
    _derivative!,
    _gradient,
    _gradient!,
    _hessian,
    _hessian_vector_product,
    _jacobian

using FiniteDifferences
using LinearAlgebra: Diagonal, dot

@testset "Differentiation backend" begin
    fd51 = Manifolds.FiniteDifferencesBackend()
    @testset "diff_backend" begin
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend
        @test length(diff_backends()) == 1
        @test diff_backends()[1] isa Manifolds.FiniteDifferencesBackend

        @test fd51.method.p == 5
        @test fd51.method.q == 1
        fd71 = Manifolds.FiniteDifferencesBackend(central_fdm(7, 1))
        @test diff_backend!(fd71) == fd71
        @test diff_backend() == fd71
    end

    using ForwardDiff

    fwd_diff = Manifolds.ForwardDiffBackend()
    @testset "ForwardDiff" begin
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend
        @test length(diff_backends()) == 2
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
        @test length(diff_backends()) == 3
        @test diff_backends()[3] == finite_diff

        @test diff_backend!(finite_diff) == finite_diff
        @test diff_backend() == finite_diff
        @test diff_backend!(fd51) isa Manifolds.FiniteDifferencesBackend
        @test diff_backend() isa Manifolds.FiniteDifferencesBackend

        diff_backend!(finite_diff)
        @test diff_backend() == finite_diff
        diff_backend!(fd51)
    end

    @testset "gradient/jacobian/hessian" begin
        diff_backend!(fd51)
        r2 = Euclidean(2)

        c1 = FunctionCurve(r2) do t
            return [sin(t), cos(t)]
        end
        f1 = FunctionRealField(r2) do x
            return x[1] + x[2]^2
        end
        f2 = FunctionRealField(r2) do x
            return 3 * x[1] * x[2] + x[2]^3
        end

        @testset "Inference" begin
            v = [-1.0, -1.0]
            @test (@inferred _derivative(c1, 0.0, Manifolds.ForwardDiffBackend())) ≈
                  [1.0, 0.0]
            @test (@inferred _derivative!(c1, v, 0.0, Manifolds.ForwardDiffBackend())) === v
            @test v ≈ [1.0, 0.0]

            @test (@inferred _derivative(c1, 0.0, finite_diff)) ≈ [1.0, 0.0]
            @test (@inferred _gradient(f1, [1.0, -1.0], finite_diff)) ≈ [1.0, -2.0]
        end

        @testset for backend in [fd51, fwd_diff, finite_diff]
            diff_backend!(backend)
            @test _derivative(c1, 0.0) ≈ [1.0, 0.0]
            v = [-1.0, -1.0]
            @test _derivative!(c1, v, 0.0) === v
            @test isapprox(v, [1.0, 0.0])
            @test _gradient(f1, [1.0, -1.0]) ≈ [1.0, -2.0]
            @test _gradient!(f1, v, [1.0, -1.0]) === v
            @test v ≈ [1.0, -2.0]
            Hp = [0.0 3.0; 3.0 -6.0]
            @test _hessian(f2, [1.0, -1.0]) ≈ Hp atol = 1e-5
            @test _hessian_vector_product(f2, [1.0, -1.0], [2.0, -1.0]) ≈
                Hp * [2.0, -1.0] atol = 1e-5
            @test _hessian_vector_product(f2, [1.0, -1.0], [-2.0, 3.0]) ≈
                Hp * [-2.0, 3.0] atol = 1e-5
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

rb_proj = Manifolds.RiemannianProjectionDiffBackend(diff_backend())

@testset "Riemannian differentials" begin
    s2 = Sphere(2)
    p = [0.0, 0.0, 1.0]
    q = [1.0, 0.0, 0.0]
    c1 = FunctionCurve(s2) do t
        return geodesic(s2, q, p, t)
    end
    @test domain(c1) === ℝ

    Xval = [-sqrt(2) / 2, 0.0, sqrt(2) / 2]
    @test isapprox(s2, c1(π / 4), r_differential(c1, π / 4), Xval)
    X = similar(p)
    r_differential!(c1, X, π / 4)
    @test isapprox(s2, c1(π / 4), X, Xval)

    @testset for backend in [rb_onb_fd51, rb_onb_fwd_diff, rb_onb_finite_diff]
        @test isapprox(s2, c1(π / 4), r_differential(c1, π / 4, backend), Xval)
        X = similar(p)
        r_differential!(c1, X, π / 4, backend)
        @test isapprox(s2, c1(π / 4), X, Xval)
    end
end

@testset "Riemannian gradients and hessians" begin
    s2 = Sphere(2)
    f1 = FunctionRealField(s2) do p
        return p[1]
    end
    @test codomain(f1) === ℝ

    q = [sqrt(2) / 2, 0, sqrt(2) / 2]
    @test isapprox(s2, q, r_gradient(f1, q), [0.5, 0.0, -0.5])
    for backend in [rb_onb_default, rb_proj]
        @test isapprox(s2, q, r_gradient(f1, q, backend), [0.5, 0.0, -0.5])
    end
    X = similar(q)
    r_gradient!(f1, X, q)
    @test isapprox(s2, q, X, [0.5, 0.0, -0.5])
    for backend in [rb_onb_default, rb_proj]
        r_gradient!(f1, X, q, backend)
        @test isapprox(s2, q, X, [0.5, 0.0, -0.5])
    end

    Hp = [-sqrt(2) / 2 0.0; 0.0 -sqrt(2) / 2]
    @test r_hessian(f1, q) ≈ Hp atol = 1e-6
    X1 = [-1.0, 1.0, 1.0]
    X2 = [1.0, 3.0, -1.0]
    basis = DefaultOrthonormalBasis()
    @test r_hessian_vector_product(f1, q, X1) ≈
        get_vector(s2, q, Hp * get_coordinates(s2, q, X1, basis), basis) atol = 1e-6
    @test r_hessian_vector_product(f1, q, X2) ≈
        get_vector(s2, q, Hp * get_coordinates(s2, q, X2, basis), basis) atol = 1e-6
    for backend in [rb_onb_default2]
        @test r_hessian(f1, q, backend) ≈ Hp atol = 1e-6
        @test r_hessian_vector_product(f1, q, X1, backend) ≈
            get_vector(s2, q, Hp * get_coordinates(s2, q, X1, basis), basis) atol = 1e-6
        @test r_hessian_vector_product(f1, q, X2, backend) ≈
            get_vector(s2, q, Hp * get_coordinates(s2, q, X2, basis), basis) atol = 1e-6
    end
end
