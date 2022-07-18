include("../utils.jl")
include("group_utils.jl")

@testset "General Unitary Groups" begin
    @testset "Orthogonal Group" begin
        O2 = Orthogonal(2)
        @test repr(O2) == "Orthogonal(2)"

        for n in [2, 3, 4, 5] # 2-4 have special implementations, 5 for generic case
            On = Orthogonal(n)
            X = zeros(n, n)
            X[1, 2] = 1.0
            X[2, 1] = -1.0
            p = exp_lie(On, X)
            X2 = log_lie(On, p)
            e = Identity(MultiplicationOperation())
            # They are not yet inverting, p2 is on the “other half”
            # but taking the log again should also be X ahain
            @test isapprox(On, e, X2, X)
            @test log_lie(On, e) == zeros(n, n)
        end
    end

    @testset "Unitary Group" begin
        U2 = Unitary(2)
        @test repr(U2) == "Unitary(2)"
        @test injectivity_radius(U2) == π

        for n in [1, 2]
            Un = Unitary(n)
            X = zeros(ComplexF64, n, n)
            X[1] = 1im
            p = 1im * Matrix{Float64}(I, n, n)
            q = exp(Un, p, X)
            X2 = log(Un, p, q)
            @test isapprox(Un, p, X, X2)
            @test inv(Un, p) == adjoint(p)
            @test injectivity_radius(Un, p) == π
        end
    end

    @testset "Special Unitary Group" begin
        SU2 = SpecialUnitary(2)
        @test repr(SU2) == "SpecialUnitary(2)"

        p = ones(2, 2)
        q = project(SU2, p)
        @test is_point(SU2, q, true)
        q2 = allocate(q)
        project!(SU2, q2, p)
        @test q == q2
        p2 = copy(p)
        p2[1, 1] = -1
        q2 = project(SU2, p2)
        @test is_point(SU2, q2, true)
        p3 = [2.0 0; 0.0 2.0] #real pos determinant
        @test project(SU2, p3) == p3 ./ 2
        Xe = ones(2, 2)
        X = project(SU2, q, Xe)
        @test is_vector(SU2, q, X)
        @test_throws ManifoldDomainError is_vector(SU2, p, X, true, true) # base point wrong
        @test_throws DomainError is_vector(SU2, q, Xe, true, true) # Xe not skew hermitian
        @test_throws DomainError is_vector(
            SU2,
            Identity(AdditionOperation()),
            Xe,
            true,
            true,
        ) # base point wrong
        e = Identity(MultiplicationOperation())
        @test_throws DomainError is_vector(SU2, e, Xe, true, true) # Xe not skew hermitian
    end

    @testset "SO(4) and O(4) exp/log edge cases" begin
        Xs = [
            [0, 0, π, 0, 0, π],  # θ = (π, π)
            [0, 0, π, 0, 0, 0],  # θ = (π, 0)
            [0, 0, π / 2, 0, 0, π],  # θ = (π, π/2)
            [0, 0, π, 0, 0, 0] ./ 2,  # θ = (π/2, 0)
            [0, 0, π, 0, 0, π] ./ 2,  # θ = (π/2, π/2)
            [0, 0, 0, 0, 0, 0],  # θ = (0, 0)
            [0, 0, 1, 0, 0, 1] .* 1e-100, # α = β ≈ 0
            [0, 0, 1, 0, 0, 1] .* 1e-6, # α = β ⩰ 0
            [0, 0, 10, 0, 0, 1] .* 1e-6, # α ⪆ β ⩰ 0
            [0, 0, π / 4, 0, 0, π / 4 - 1e-6], # α ⪆ β > 0
        ]
        Ms = [SpecialOrthogonal(4), Orthogonal(4)]
        for Xf in Xs
            @testset "rotation vector $Xf" begin
                for M in Ms
                    X = Manifolds.hat(M, Matrix(1.0I, 4, 4), Xf)
                    p = exp(X)
                    @test p ≈ exp_lie(M, X)
                    p2 = exp_lie(M, log_lie(M, p))
                    @test isapprox(M, p, p2; atol=1e-6)
                end
            end
        end
    end
end
