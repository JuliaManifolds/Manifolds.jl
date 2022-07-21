include("../utils.jl")

using Quaternions

@testset "Orthogonal Matrices" begin
    M = OrthogonalMatrices(3)
    @test repr(M) == "OrthogonalMatrices(3)"
    @test injectivity_radius(M, PolarRetraction()) == π / sqrt(2.0)
    p = project(M, ones(3, 3))
    @test is_point(M, p, true)
end

@testset "Unitary Matrices" begin
    M = UnitaryMatrices(2)
    @test repr(M) == "UnitaryMatrices(2)"

    # wrong length of size
    @test_throws DomainError is_point(M, zeros(1), true)
    @test_throws DomainError is_point(M, zeros(3, 3), true)
    pF = 1 / 2 .* [1im 1im; -1im 1im]
    @test_throws DomainError is_point(M, pF, true)
    # Determinant not one
    pF2 = [1im 1.0; 0.0 -1im]
    @test_throws DomainError is_point(M, pF2, true)
    p = [1im 0.0; 0.0 1im]
    @test is_point(M, p, true)

    @test_throws DomainError is_vector(M, p, zeros(1), true)
    @test_throws DomainError is_vector(M, p, zeros(3, 3), true)
    # not skew
    @test_throws DomainError is_vector(M, p, ones(2, 2), true)
    X = [0.0 1.0; -1.0 0.0]
    @test is_vector(M, p, X, true)

    q = project(M, ones(2, 2))
    @test is_point(M, q, true)
    q2 = project(M, 1im * ones(2, 2))
    @test is_point(M, q2, true)

    r = exp(M, p, X)
    X2 = log(M, p, r)
    @test isapprox(M, p, X, X2)
end

@testset "Quaternionic Unitary Matrices" begin
    M = QuaternionicUnitaryMatrices(1)
    @test repr(M) == "QuaternionicUnitaryMatrices(1)"

    # wrong length of size
    @test_throws DomainError is_point(M, zeros(2, 2), true)

    # Determinant not one
    pF2 = [1im 1.0; 0.0 -1im]
    @test_throws DomainError is_point(M, pF2, true)
    p = QuaternionF64(
        0.4815296357756736,
        0.6041613272484806,
        -0.2322369798903669,
        0.5909181717450419,
        true,
    )

    @test is_point(M, [p;;])
    @test is_point(M, p)

    @test_throws DomainError is_vector(M, p, zeros(2, 2), true)
    # not skew
    @test_throws DomainError is_vector(M, p, Quaternion(1, 0, 0, 0), true)
    X = Quaternion(0.0, 0, 0, 1)
    @test is_vector(M, p, X)

    pu = QuaternionF64(
        -0.2178344173900564,
        -0.4072721993877449,
        -2.2484219560115712,
        -0.4718862793239344,
        false,
    )
    q = project(M, pu)
    @test is_point(M, q)
    @test isapprox(q, normalize(pu))
end
