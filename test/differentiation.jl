using Test
using Manifolds
using Manifolds: _derivative, _gradient, _jacobian
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

    @testset "gradient/jacobian" begin
        diff_backend!(fd51)
        r2 = Euclidean(2)

        c1 = FunctionCurve(r2) do t
            return [sin(t), cos(t)]
        end
        f1 = FunctionRealField(r2) do x
            return x[1]+x[2]^2
        end
        @testset "Inference" begin
            @test (@inferred _derivative(c1, 0.0, Manifolds.ForwardDiffBackend())) ≈ [1.0, 0.0]

            @test (@inferred _derivative(c1, 0.0, finite_diff)) ≈ [1.0, 0.0]
            @test (@inferred _gradient(f1, [1.0, -1.0], finite_diff)) ≈ [1.0, -2.0]
        end

        @testset for backend in [
            fd51,
            fwd_diff,
            finite_diff
        ]
            diff_backend!(backend)
            @test _derivative(c1, 0.0) ≈ [1.0, 0.0]
            @test _gradient(f1, [1.0, -1.0]) ≈ [1.0, -2.0]
        end
        diff_backend!(Manifolds.NoneDiffBackend())
        @testset for backend in [fd51, Manifolds.ForwardDiffBackend()]
            @test _derivative(c1, 0.0, backend) ≈ [1.0, 0.0]
            @test _gradient(f1, [1.0, -1.0], backend) ≈ [1.0, -2.0]
        end

        diff_backend!(fd51)
    end
end

@testset "Riemannian derivatives" begin
    s2 = Sphere(2)
    p = [0.0, 0.0, 1.0]
    q = [1.0, 0.0, 0.0]
    c1 = FunctionCurve(s2) do t
        return geodesic(s2, q, p, t)
    end
    @test isapprox(s2, c1(π/4), r_derivative(c1, π/4), [-sqrt(2)/2, 0.0, sqrt(2)/2])
end
