using NLsolve
using LinearAlgebra

include("utils.jl")

Random.seed!(10)

@testset "approximate inverse retractions" begin
    @testset "NLsolveInverseRetraction" begin
        @testset "constructor" begin
            X = randn(3)

            @test NLsolveInverseRetraction <: ApproximateInverseRetraction
            m1 = NLsolveInverseRetraction(ExponentialRetraction())
            @test m1.retraction === ExponentialRetraction()
            @test m1.X0 === nothing
            @test !m1.project_tangent
            @test !m1.project_point
            @test isempty(m1.nlsolve_kwargs)

            m2 = NLsolveInverseRetraction(
                PolarRetraction(),
                [1.0, 2.0, 3.0];
                project_tangent=true,
                project_point=true,
                autodiff=:forward,
            )
            @test m2.retraction === PolarRetraction()
            @test m2.X0 == [1.0, 2.0, 3.0]
            @test m2.project_tangent
            @test m2.project_point
            @test (; m2.nlsolve_kwargs...) == (; autodiff=:forward)
        end
        @testset "Euclidean" begin
            M = Euclidean(3)
            p = [1.0, 2.0, 3.0]
            q = [4.0, 5.0, 6.0]
            retr_method = ExponentialRetraction()
            inv_retr_method = NLsolveInverseRetraction(retr_method)
            X = inverse_retract(M, p, q, inv_retr_method)
            @test is_vector(M, p, X)
            @test X isa Vector{Float64}
            @test X ≈ q - p
        end
        @testset "Power" begin
            M = PowerManifold(Euclidean(2), NestedPowerRepresentation(), 2)
            p = [[1.0, 2.0], [3.0, 4.0]]
            q = [[5.0, 6.0], [7.0, 8.0]]
            retr_method = ExponentialRetraction()
            inv_retr_method = NLsolveInverseRetraction(retr_method)
            X = inverse_retract(M, p, q, inv_retr_method)
            @test is_vector(M, p, X)
            @test X isa Vector{Vector{Float64}}
            @test X ≈ q - p
        end
        @testset "Sphere" begin
            M = Sphere(2)
            p = [1.0, 0.0, 0.0]
            q = [1 / sqrt(2), 1 / sqrt(2), 0.0]
            X_exp = inverse_retract(M, p, q, ProjectionInverseRetraction())
            # vector must be nonzero to converge
            X0 = randn(3) .* eps()
            inv_retr_method =
                NLsolveInverseRetraction(ProjectionRetraction(), X0; project_point=true)
            X = inverse_retract(M, p, q, inv_retr_method)
            @test is_vector(M, p, X; atol=1e-9)
            @test X ≈ X_exp atol=1e-8
            @test_throws OutOfInjectivityRadiusError inverse_retract(
                M,
                p,
                -p,
                inv_retr_method,
            )
        end

        @testset "Circle(ℂ)" begin
            M = Circle(ℂ)
            p = [1.0 * im]
            X = [p[1] * im * (π / 4)]
            q = exp(M, p, X)
            X_exp = log(M, p, q)
            inv_retr_method =
                NLsolveInverseRetraction(ExponentialRetraction(); project_point=true)
            X = inverse_retract(M, p, q, inv_retr_method)
            @test is_vector(M, p, X; atol=1e-8)
            @test X ≈ X_exp
        end
    end
end
