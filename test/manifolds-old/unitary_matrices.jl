include("../header.jl")

using Quaternions

@testset "Orthogonal Matrices" begin
    M = OrthogonalMatrices(3)
    @test repr(M) == "OrthogonalMatrices(3)"
    @test repr(OrthogonalMatrices(3; parameter = :field)) ==
        "OrthogonalMatrices(3; parameter=:field)"
    @test injectivity_radius(M, PolarRetraction()) == π / sqrt(2.0)
    @test manifold_dimension(M) == 3
    @test injectivity_radius(M) == π * sqrt(2.0)
    @test injectivity_radius(OrthogonalMatrices(3; parameter = :field)) == π * sqrt(2.0)
    @test injectivity_radius(OrthogonalMatrices(1; parameter = :field)) == 0.0
    @test !is_flat(M)
    p = project(M, ones(3, 3))
    @test is_point(M, p; error = :error)
    @test is_point(M, rand(M); error = :error)
    @test abs(rand(OrthogonalMatrices(1))[]) == 1
    @test is_vector(M, p, rand(M; vector_at = p))
    @test is_point(M, rand(MersenneTwister(), M); error = :error)
    @test abs(rand(MersenneTwister(), OrthogonalMatrices(1))[]) == 1
    @test is_vector(M, p, rand(MersenneTwister(), M; vector_at = p))
    @test default_vector_transport_method(M) === ProjectionTransport()

    @test get_embedding(M) === Euclidean(3, 3)
    @test get_embedding(OrthogonalMatrices(3; parameter = :field)) === Euclidean(3, 3; parameter = :field)

    @test manifold_volume(OrthogonalMatrices(1)) ≈ 2
    @test manifold_volume(OrthogonalMatrices(2)) ≈ 4 * π * sqrt(2)
    @test manifold_volume(OrthogonalMatrices(3)) ≈ 16 * π^2 * sqrt(2)
    @test manifold_volume(OrthogonalMatrices(4)) ≈ 2 * (2 * π)^4 * sqrt(2)
    @test manifold_volume(OrthogonalMatrices(5)) ≈ 8 * (2 * π)^6 / 6 * sqrt(2)
end

@testset "Unitary Matrices" begin
    M = UnitaryMatrices(2)
    @test repr(M) == "UnitaryMatrices(2)"
    @test repr(UnitaryMatrices(2; parameter = :field)) ==
        "UnitaryMatrices(2; parameter=:field)"
    @test manifold_dimension(M) == 4
    @test !is_flat(M)
    @test injectivity_radius(M) == π

    # wrong length of size
    @test_throws DomainError is_point(M, zeros(1); error = :error)
    @test_throws DomainError is_point(M, zeros(3, 3); error = :error)
    pF = 1 / 2 .* [1im 1im; -1im 1im]
    @test_throws DomainError is_point(M, pF; error = :error)
    # Determinant not one
    pF2 = [1im 1.0; 0.0 -1im]
    @test_throws DomainError is_point(M, pF2; error = :error)
    p = [1im 0.0; 0.0 1im]
    @test is_point(M, p; error = :error)

    @test_throws DomainError is_vector(M, p, zeros(1); error = :error)
    @test_throws DomainError is_vector(M, p, zeros(3, 3); error = :error)
    # not skew
    @test_throws DomainError is_vector(M, p, ones(2, 2); error = :error)
    X = [0.0 1.0; -1.0 0.0]
    @test is_vector(M, p, X; error = :error)

    q = project(M, ones(2, 2))
    @test is_point(M, q; error = :error)
    q2 = project(M, 1im * ones(2, 2))
    @test is_point(M, q2; error = :error)

    r = exp(M, p, X)
    X2 = log(M, p, r)
    @test isapprox(M, p, X, X2)
    r1 = Manifolds.exp_fused(M, p, X, 1.0)
    @test isapprox(M, r, r1; atol = 1.0e-10)

    @testset "Projection" begin
        M = UnitaryMatrices(2)
        pE = [2im 0.0; 0.0 2im]
        p = project(M, pE)
        @test is_point(M, p; error = :error)
        pE[2, 1] = 1.0
        X = project(M, p, pE)
        @test is_vector(M, p, X; error = :error)
    end
    @testset "Random" begin
        M = UnitaryMatrices(2)
        Random.seed!(23)
        p = rand(M)
        @test is_point(M, p; error = :error)
        @test is_vector(M, p, rand(M; vector_at = p); error = :error)
    end
    @testset "Riemannian Hessian" begin
        p = Matrix{Float64}(I, 2, 2)
        X = [0.0 3.0; -3.0 0.0]
        V = [1.0 0.0; 1.0 0.0]
        @test Weingarten(M, p, X, V) == -1 / 2 * p * (V' * X - X' * V)
        G = [0.0 1.0; 0.0 0.0]
        H = [0.0 0.0; 2.0 0.0]
        @test riemannian_Hessian(M, p, G, H, X) == [0.0 -1.0; 1.0 0.0]
    end
    @test !is_flat(M)
    @test is_flat(UnitaryMatrices(1))
    @test is_flat(UnitaryMatrices(1; parameter = :field))

    @testset "manifold_volume" begin
        @test manifold_volume(UnitaryMatrices(1)) ≈ 2 * π
        @test manifold_volume(UnitaryMatrices(2)) ≈ 4 * π^3
        @test manifold_volume(UnitaryMatrices(3)) ≈ sqrt(3) * 2 * π^6
        @test manifold_volume(UnitaryMatrices(4)) ≈ sqrt(2) * 8 * π^10 / 12
    end
end

@testset "Special unitary matrices" begin
    @test manifold_volume(SpecialUnitaryMatrices(1)) ≈ 1
    @test manifold_volume(SpecialUnitaryMatrices(2)) ≈ 2 * π^2
    @test manifold_volume(SpecialUnitaryMatrices(3)) ≈ sqrt(3) * π^5
    @test manifold_volume(SpecialUnitaryMatrices(4)) ≈ sqrt(2) * 4 * π^9 / 12

    @test manifold_dimension(SpecialUnitaryMatrices(2)) == 3
    @test manifold_dimension(SpecialUnitaryMatrices(3)) == 8
    @test manifold_dimension(SpecialUnitaryMatrices(4)) == 15

    @test injectivity_radius(SpecialUnitaryMatrices(2)) == π * sqrt(2)
    @test injectivity_radius(SpecialUnitaryMatrices(3)) == π * sqrt(2)
    @test injectivity_radius(SpecialUnitaryMatrices(4)) == π * sqrt(2)
end

@testset "Quaternionic Unitary Matrices" begin
    M = UnitaryMatrices(1, ℍ)
    @test repr(M) == "UnitaryMatrices(1, ℍ)"
    @test repr(UnitaryMatrices(1, ℍ; parameter = :field)) ==
        "UnitaryMatrices(1, ℍ; parameter=:field)"
    @test manifold_dimension(M) == 3
    @test injectivity_radius(M) == π
    @test !is_flat(M)
    @testset "rand" begin
        p = rand(M)
        @test is_point(M, p)
        X = rand(M; vector_at = p)
        @test is_vector(M, p, X)
        p = rand(MersenneTwister(), M)
        @test is_point(M, p)
        X = rand(MersenneTwister(), M; vector_at = p)
        @test is_vector(M, p, X)
    end
    @testset "Exp/Log and issapprox" begin
        p = QuaternionF64(1.0, 0.0, 0.0, 0.0)
        X = QuaternionF64(0.1, 0.2, 0.2, 0.3)
        q = exp(M, p, X)
        q2 = Manifolds.exp_fused(M, p, X, 1.0)
        @test isapprox(M, q, q2)
        X2 = log(M, p, q)
        @test isapprox(M, p, X, X2)
    end

    # wrong length of size
    @test_throws DomainError is_point(M, zeros(2, 2); error = :error)

    # Determinant not one
    pF2 = [quat(0, 1, 0, 0) 1.0; 0.0 -quat(0, 1, 0, 0)]
    @test_throws DomainError is_point(M, pF2; error = :error)
    p = QuaternionF64(
        0.4815296357756736,
        0.6041613272484806,
        -0.2322369798903669,
        0.5909181717450419,
    )

    @test is_point(M, fill(p, 1, 1))
    @test is_point(M, p)

    @test_throws DomainError is_vector(M, p, zeros(2, 2); error = :error)
    # not skew
    @test_throws DomainError is_vector(M, p, Quaternion(1, 0, 0, 0); error = :error)
    X = Quaternion(0.0, 0, 0, 1)
    @test is_vector(M, p, X)

    pu = QuaternionF64(
        -0.2178344173900564,
        -0.4072721993877449,
        -2.2484219560115712,
        -0.4718862793239344,
    )
    q = project(M, pu)
    @test is_point(M, q)
    @test isapprox(q, sign(pu))

    @test get_coordinates(M, p, Quaternion(0, 1, 2, 3), DefaultOrthonormalBasis(ℝ)) ==
        SA[1, 2, 3]
    @test get_vector(M, p, SA[1, 2, 3], DefaultOrthonormalBasis(ℝ)) ==
        Quaternion(0, 1, 2, 3)
    @test number_of_coordinates(M, DefaultOrthonormalBasis(ℍ)) == 3

    @test get_basis(M, p, DefaultOrthonormalBasis(ℝ)).data == [
        Quaternion(0.0, 1.0, 0.0, 0.0),
        Quaternion(0.0, 0.0, 1.0, 0.0),
        Quaternion(0.0, 0.0, 0.0, 1.0),
    ]
end

@testset "SO(4) and O(4) exp/log edge cases" begin
    Xs = [
        [0, 0, π, 0, 0, π],  # θ = (π, π)
        [0, 0, π, 0, 0, 0],  # θ = (π, 0)
        [0, 0, π / 2, 0, 0, π],  # θ = (π, π/2)
        [0, 0, π, 0, 0, 0] ./ 2,  # θ = (π/2, 0)
        [0, 0, π, 0, 0, π] ./ 2,  # θ = (π/2, π/2)
        [0, 0, 0, 0, 0, 0],  # θ = (0, 0)
        [0, 0, 1, 0, 0, 1] .* 1.0e-100, # α = β ≈ 0
        [0, 0, 1, 0, 0, 1] .* 1.0e-6, # α = β ⩰ 0
        [0, 0, 10, 0, 0, 1] .* 1.0e-6, # α ⪆ β ⩰ 0
        [0, 0, π / 4, 0, 0, π / 4 - 1.0e-6], # α ⪆ β > 0
    ]
    Ms = [Rotations(4), OrthogonalMatrices(4)]
    E = diagm(ones(4))
    for Xf in Xs
        for M in Ms
            @testset "for $Xf on $M" begin
                X = get_vector(M, E, Xf, DefaultOrthogonalBasis())
                p = exp(X)
                @test p ≈ exp(M, E, X)
                p3 = exp(M, E, log(M, E, p))
                # broken for 9 of the 10
                @test isapprox(M, p, p3; atol = 1.0e-4)
            end
        end
    end
    R1 = diagm([-1.0, -1.0, 1.0, 1.0])
    X1a = log(Rotations(4), E, R1)
    @test is_vector(Rotations(4), E, X1a)
    @test X1a[1, 2] ≈ π

    R2 = diagm([-1.0, 1.0, -1.0, 1.0])
    X2a = log(Rotations(4), E, R2)
    @test is_vector(Rotations(4), E, X2a)
    @test X2a[1, 3] ≈ π

    R3 = diagm([1.0, -1.0, -1.0, 1.0])
    X3a = log(Rotations(4), E, R3)
    @test is_vector(Rotations(4), E, X3a)
    @test X3a[2, 3] ≈ π
end
