include("../header.jl")
include("group_utils.jl")

@testset "General Unitary Groups" begin
    # SpecialUnitary -> injectivity us also π√2
    SU3 = SpecialUnitary(3)
    @test injectivity_radius(SU3) == π * √2
    @testset "Orthogonal Group" begin
        O2 = Orthogonal(2)
        @test repr(O2) == "Orthogonal(2)"

        for n in [2, 3, 4, 5] # 2-4 have special implementations, 5 for generic case
            On = Orthogonal(n)
            injectivity_radius(On) ≈ π * √2
            X = zeros(n, n)
            X[1, 2] = 1.0
            X[2, 1] = -1.0
            p = exp_lie(On, X)
            X2 = log_lie(On, p)
            e = Identity(MultiplicationOperation())
            # They are not yet inverting, p2 is on the “other half”
            # but taking the log again should also be X again
            @test isapprox(On, e, X2, X)
            @test log_lie(On, e) == zeros(n, n)
        end

        @test manifold_volume(Orthogonal(1)) ≈ 2
        @test manifold_volume(Orthogonal(2)) ≈ 4 * π * sqrt(2)
        @test manifold_volume(Orthogonal(3)) ≈ 16 * π^2 * sqrt(2)
        @test manifold_volume(Orthogonal(4)) ≈ 2 * (2 * π)^4 * sqrt(2)
        @test manifold_volume(Orthogonal(5)) ≈ 8 * (2 * π)^6 / 6 * sqrt(2)
    end

    @testset "Special Orthogonal Group" begin
        @test manifold_volume(SpecialOrthogonal(1)) ≈ 1
        @test manifold_volume(SpecialOrthogonal(2)) ≈ 2 * π * sqrt(2)
        @test manifold_volume(SpecialOrthogonal(3)) ≈ 8 * π^2 * sqrt(2)
        @test manifold_volume(SpecialOrthogonal(4)) ≈ (2 * π)^4 * sqrt(2)
        @test manifold_volume(SpecialOrthogonal(5)) ≈ 4 * (2 * π)^6 / 6 * sqrt(2)

        M = SpecialOrthogonal(2)
        p = [
            0.22632098602578843 0.9740527764368391
            -0.9740527764368391 0.22632098602578843
        ]
        X = [0.0 -0.7071067811865475; 0.7071067811865475 0.0]
        @test volume_density(M, p, X) ≈ 1.0

        @testset "SO(2) Lie Bracket == 0" begin
            Y = [0.0 0.7071067811865475; -0.7071067811865475 0.0]
            X_ = copy(X)
            X_[1, 2] += 1e-16
            @test is_vector(M, identity_element(M), X_)
            @test lie_bracket(M, X_, Y) == zeros(2, 2)
            @test lie_bracket!(M, similar(X_), X_, Y) == zeros(2, 2)
        end

        M = SpecialOrthogonal(3)
        p = [
            -0.5908399013383766 -0.6241917041179139 0.5111681988316876
            -0.7261666986267721 0.13535732881097293 -0.6740625485388226
            0.35155388888753836 -0.7694563730631729 -0.5332417398896261
        ]
        X = [
            0.0 -0.30777760628130063 0.5499897386953444
            0.30777760628130063 0.0 -0.32059980100053004
            -0.5499897386953444 0.32059980100053004 0.0
        ]
        @test volume_density(M, p, X) ≈ 0.8440563052346255
        @test volume_density(M, p, zero(X)) ≈ 1.0

        M = SpecialOrthogonal(4)
        p = [
            -0.09091199873970474 -0.5676546886791307 -0.006808638869334249 0.8182034009599919
            -0.8001176365300662 0.3161567169523502 -0.4938592872334223 0.12633171594159726
            -0.5890394255366699 -0.2597679221590146 0.7267279425385695 -0.23962403743004465
            -0.0676707570677516 -0.7143764493344514 -0.4774129704812182 -0.5071132150619608
        ]
        X = [
            0.0 0.2554704296965055 0.26356215573144676 -0.4070678736115306
            -0.2554704296965055 0.0 -0.04594199053786204 -0.10586374034761421
            -0.26356215573144676 0.04594199053786204 0.0 0.43156436122007846
            0.4070678736115306 0.10586374034761421 -0.43156436122007846 0.0
        ]
        @test volume_density(M, p, X) ≈ 0.710713830700454

        # random Lie algebra element
        @test is_vector(M, Identity(M), rand(M; vector_at=Identity(M)))
    end

    @testset "Unitary Group" begin
        U2 = Unitary(2)
        @test repr(U2) == "Unitary(2)"
        @test injectivity_radius(U2) == π

        @test manifold_volume(Unitary(1)) ≈ 2 * π
        @test manifold_volume(Unitary(2)) ≈ 4 * π^3
        @test manifold_volume(Unitary(3)) ≈ sqrt(3) * 2 * π^6
        @test manifold_volume(Unitary(4)) ≈ sqrt(2) * 8 * π^10 / 12

        @test identity_element(U2) isa Matrix{ComplexF64}

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
        )
        X = Quaternion(0.0, 0, 0, 1)
        q = exp(QU1, p, X)
        X2 = log(QU1, p, q)
        @test isapprox(QU1, p, X, X2)
        q2 = exp_lie(QU1, X)
        X3 = log_lie(QU1, q2)
        @test isapprox(QU1, p, X, X3)
        q3 = Manifolds.exp_fused(QU1, p, X, 1.0)
        @test isapprox(QU1, q, q3)

        q3 = Ref(Quaternion(0.0))
        exp_lie!(QU1, q3, X)
        @test isapprox(QU1, q3[], q2)
        X4 = Ref(Quaternion(0.0))
        log_lie!(QU1, X4, q2)
        @test isapprox(QU1, identity_element(QU1), X3, X4[])

        X5 = fill(Quaternion(0.0), 1, 1)
        log_lie!(QU1, X5, fill(q2, 1, 1))
        @test isapprox(QU1, identity_element(QU1), X3, X5[])

        @test inv(QU1, p) == conj(p)

        @test project(QU1, p, Quaternion(1.0, 2.0, 3.0, 4.0)) ===
              Quaternion(0.0, 2.0, 3.0, 4.0)
    end

    @testset "Special Unitary Group" begin
        SU2 = SpecialUnitary(2)
        @test repr(SU2) == "SpecialUnitary(2)"

        p = ones(2, 2)
        q = project(SU2, p)
        @test is_point(SU2, q; error=:error)
        q2 = allocate(q)
        project!(SU2, q2, p)
        @test q == q2
        p2 = copy(p)
        p2[1, 1] = -1
        q2 = project(SU2, p2)
        @test is_point(SU2, q2; error=:error)
        p3 = [2.0 0; 0.0 2.0] #real pos determinant
        @test project(SU2, p3) == p3 ./ 2
        Xe = ones(2, 2)
        X = project(SU2, q, Xe)
        @test is_vector(SU2, q, X)
        @test_throws DomainError is_vector(SU2, p, X, true; error=:error) # base point wrong
        @test_throws DomainError is_vector(SU2, q, Xe, true; error=:error) # Xe not skew hermitian
        @test_throws DomainError is_vector(
            SU2,
            Identity(AdditionOperation()),
            Xe,
            true;
            error=:error,
        ) # base point wrong
        e = Identity(MultiplicationOperation())
        @test_throws DomainError is_vector(SU2, e, Xe, true; error=:error) # Xe not skew hermitian

        @test manifold_volume(SpecialUnitary(1)) ≈ 1
        @test manifold_volume(SpecialUnitary(2)) ≈ 2 * π^2
        @test manifold_volume(SpecialUnitary(3)) ≈ sqrt(3) * π^5
        @test manifold_volume(SpecialUnitary(4)) ≈ sqrt(2) * 4 * π^9 / 12
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

    @testset "field parameter" begin
        G = Orthogonal(2; parameter=:field)
        @test repr(G) == "Orthogonal(2; parameter=:field)"

        SU3 = SpecialUnitary(3; parameter=:field)
        @test repr(SU3) == "SpecialUnitary(3; parameter=:field)"

        G = Unitary(3, ℂ; parameter=:field)
        @test repr(G) == "Unitary(3; parameter=:field)"

        G = Unitary(3, ℍ; parameter=:field)
        @test repr(G) == "Unitary(3, ℍ; parameter=:field)"
    end
end
