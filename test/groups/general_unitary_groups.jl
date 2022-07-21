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

        for n in [1, 2, 3]
            Un = Unitary(n)
            X = zeros(ComplexF64, n, n)
            X[1] = 1im
            p = 1im * Matrix{Float64}(I, n, n)
            q = exp(Un, p, X)
            X2 = log(Un, p, q)
            @test isapprox(Un, p, X, X2)
            q2 = exp_lie(Un, X)
            X3 = log_lie(Un, q2)
            @test isapprox(Un, p, X, X3)
            @test inv(Un, p) == adjoint(p)
            @test injectivity_radius(Un, p) == π
        end
    end

    @testset "Quaternionic Unitary Group" begin
        QU1 = Unitary(1, ℍ)
        @test repr(QU1) == "Unitary(1, ℍ)"

        @test identity_element(QU1) === Quaternion(1.0)

        p = QuaternionF64(
            0.4815296357756736,
            0.6041613272484806,
            -0.2322369798903669,
            0.5909181717450419,
            true,
        )
        X = Quaternion(0.0, 0, 0, 1)
        q = exp(QU1, p, X)
        X2 = log(QU1, p, q)
        @test isapprox(QU1, p, X, X2)
        q2 = exp_lie(QU1, X)
        X3 = log_lie(QU1, q2)
        @test isapprox(QU1, p, X, X3)

        q3 = Ref(Quaternion(0.0))
        exp_lie!(QU1, q3, X)
        @test isapprox(QU1, q3[], q2)
        X4 = Ref(Quaternion(0.0))
        log_lie!(QU1, X4, q2)
        @test isapprox(QU1, identity_element(QU1), X3, X4[])

        @test inv(QU1, p) == conj(p)

        @test project(QU1, p, Quaternion(1.0, 2.0, 3.0, 4.0)) ===
              Quaternion(0.0, 2.0, 3.0, 4.0)
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
            for M in Ms
                @testset "for $Xf on $M" begin
                    X = Manifolds.hat(M, Matrix(1.0I, 4, 4), Xf)
                    p = exp(X)
                    @test p ≈ exp_lie(M, X)
                    p2 = exp_lie(M, log_lie(M, p))
                    @test isapprox(M, p, p2; atol=1e-6)
                    # pass through to the manifold (Orthogonal / Rotations)
                    @test p ≈ exp(M, one(p), X)
                    p3 = exp(M, one(p), log(M, one(p), p))
                    # broken for 9 of the 10
                    @test isapprox(M, p, p3; atol=1e-4)
                end
            end
        end
        E = diagm(ones(4))
        R1 = diagm([-1.0, -1.0, 1.0, 1.0])
        X1a = log(Rotations(4), E, R1)
        X1b = log_lie(SpecialOrthogonal(4), R1)
        @test isapprox(X1a, X1b)
        @test is_vector(Rotations(4), E, X1b)
        @test X1a[1, 2] ≈ π

        R2 = diagm([-1.0, 1.0, -1.0, 1.0])
        X2a = log(Rotations(4), E, R2)
        X2b = log_lie(SpecialOrthogonal(4), R2)
        @test isapprox(X2a, X2b)
        @test is_vector(Rotations(4), E, X2b)
        @test X2a[1, 3] ≈ π

        R3 = diagm([1.0, -1.0, -1.0, 1.0])
        X3a = log(Rotations(4), E, R3)
        X3b = log_lie(SpecialOrthogonal(4), R3)
        @test isapprox(X3a, X3b)
        @test is_vector(Rotations(4), E, X3b)
        @test X3a[2, 3] ≈ π
    end
end
