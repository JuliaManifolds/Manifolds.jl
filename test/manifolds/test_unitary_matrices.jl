using LinearAlgebra, Manifolds, Quaternions, Test, ManifoldsBase, StaticArrays

@testset "Unitray Matrices" begin
    M = UnitaryMatrices(2)
    p1 = [1im 0.0; 0.0 1im]
    X1 = [0.0 1.0; -1.0 0.0]
    p2 = project(M, ones(2, 2))
    p3 = project(M, 1im * ones(2, 2))

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                default_vector_transport_method,
                get_embedding,
                is_flat,
                project,
                rand, repr,
                # Weingarten, # TODO: V is not normal?!
                riemannian_Hessian,
            ],
            :EmbeddedPoints => [ones(2, 2), 1im .* ones(2, 2), [2im 0.0; 0.0 2im]],
            :EmbeddedVectors => [[2im 0.0; 1.0 2im]],
            :NormalVectors => [[1.0 0.0; 1.0 0.0]],
            :Points => [p1],
            :Vectors => [X1],
            :InvalidPoints => [zeros(1), zeros(3, 3), 1 / 2 .* [1im 1im; -1im 1im], [1im 1.0; 0.0 -1im]],
            :InvalidVectors => [zeros(1), zeros(3, 3), ones(2, 2)],
            :RetractionMethods => [PolarRetraction()],
        ),
        Dict(
            default_vector_transport_method => ProjectionTransport(),
            get_embedding => Euclidean(2, 2; field = ℂ),
            is_flat => false,
            manifold_dimension => 4,
            repr => "UnitaryMatrices(2)",
            injectivity_radius => π,
        )
    )

    @testset "Field parameter" begin
        MF = UnitaryMatrices(2; parameter = :field)
        @test repr(MF) ==
            "UnitaryMatrices(2; parameter=:field)"

        @test get_embedding(MF) === Euclidean(2, 2; field = ℂ, parameter = :field)
    end

    @testset "Fused exp" begin
        r = exp(M, p1, X1)
        X2 = log(M, p1, r)
        @test isapprox(M, p1, X1, X2)
        r1 = Manifolds.exp_fused(M, p1, X1, 1.0)
        @test isapprox(M, r, r1; atol = 1.0e-10)
    end

    @testset "Projection with points outside of the manifold" begin
        M = UnitaryMatrices(2)
        pE = [2im 0.0; 0.0 2im]
        p = project(M, pE)
        @test is_point(M, p; error = :error)
        pE[2, 1] = 1.0
        X = project(M, p, pE)
        @test is_vector(M, p, X; error = :error)
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

    @test is_flat(UnitaryMatrices(1))
    @test is_flat(UnitaryMatrices(1; parameter = :field))

    @testset "manifold_volume" begin
        @test manifold_volume(UnitaryMatrices(1)) ≈ 2 * π
        @test manifold_volume(UnitaryMatrices(2)) ≈ 4 * π^3
        @test manifold_volume(UnitaryMatrices(3)) ≈ sqrt(3) * 2 * π^6
        @test manifold_volume(UnitaryMatrices(4)) ≈ sqrt(2) * 8 * π^10 / 12
    end

    @testset "Polar retraction" begin
        # Test that check_det is not passed to project in UnitaryMatrices
        M = UnitaryMatrices(8)
        t = 0.3
        p = rand(M)
        X = 0.25 .* rand(M; vector_at = p)
        q = similar(p)

        ManifoldsBase.retract_fused!(M, q, p, X, t, PolarRetraction())
        @test is_point(M, q)
    end

end

@testset "Quaternionic unitary matrices" begin
    MH = UnitaryMatrices(1, ℍ)
    pH = QuaternionF64(1.0, 0.0, 0.0, 0.0)
    pH2 = QuaternionF64(0.0, 1.0, 0.0, 0.0)
    XH = QuaternionF64(0.0, 0.2, 0.2, 0.3)
    pHF = [quat(0, 1, 0, 0) 1.0; 0.0 -quat(0, 1, 0, 0)]
    PHE = QuaternionF64(-0.2178344173900564, -0.4072721993877449, -2.2484219560115712, -0.4718862793239344)
    Manifolds.Test.test_manifold(
        MH,
        Dict(
            :Functions => [
                default_vector_transport_method,
                exp,
                get_embedding,
                injectivity_radius, is_flat,
                log,
                project,
                rand, repr,
            ],
            :EmbeddedPoints => [PHE],
            :Points => [pH, pH2],
            :Vectors => [XH],
            :InvalidPoints => [pHF, zeros(2, 2)],
            :InvalidVectors => [Quaternion(1, 0, 0, 0), zeros(2, 2)],
            :Mutating => false,
        ),
        Dict(
            default_vector_transport_method => ProjectionTransport(),
            get_embedding => Euclidean(1, 1; field = ℍ),
            is_flat => false,
            manifold_dimension => 3,
            repr => "UnitaryMatrices(1, ℍ)",
            injectivity_radius => π,
        )
    )

    @testset "Field parameter" begin
        MHF = UnitaryMatrices(1, ℍ; parameter = :field)
        @test repr(MHF) ==
            "UnitaryMatrices(1, ℍ; parameter=:field)"

        @test get_embedding(MHF) === Euclidean(1, 1; field = ℍ, parameter = :field)
    end

    @testset "Fused exp" begin
        p = QuaternionF64(1.0, 0.0, 0.0, 0.0)
        X = QuaternionF64(0.1, 0.2, 0.2, 0.3)
        q = exp(MH, p, X)
        q2 = Manifolds.exp_fused(MH, p, X, 1.0)
        @test isapprox(MH, q, q2)
        # test that log also works fine
        X2 = log(M, p, q)
        @test isapprox(M, p, X, X2)
    end

    @testset "Matrix point" begin
        p = QuaternionF64(
            0.4815296357756736,
            0.6041613272484806,
            -0.2322369798903669,
            0.5909181717450419,
        )

        @test is_point(MH, fill(p, 1, 1))
        @test is_point(MH, p)
    end

    @testset "Functions with specific expectation" begin
        p = QuaternionF64(
            0.4815296357756736,
            0.6041613272484806,
            -0.2322369798903669,
            0.5909181717450419,
        )
        pu = QuaternionF64(
            -0.2178344173900564,
            -0.4072721993877449,
            -2.2484219560115712,
            -0.4718862793239344,
        )
        q = project(MH, pu)
        @test is_point(MH, q)
        @test isapprox(q, sign(pu))

        @test get_coordinates(MH, p, Quaternion(0, 1, 2, 3), DefaultOrthonormalBasis(ℝ)) ==
            SA[1, 2, 3]
        @test get_vector(MH, p, SA[1, 2, 3], DefaultOrthonormalBasis(ℝ)) ==
            Quaternion(0, 1, 2, 3)
        @test number_of_coordinates(MH, DefaultOrthonormalBasis(ℍ)) == 3

        @test get_basis(MH, p, DefaultOrthonormalBasis(ℝ)).data == [
            Quaternion(0.0, 1.0, 0.0, 0.0),
            Quaternion(0.0, 0.0, 1.0, 0.0),
            Quaternion(0.0, 0.0, 0.0, 1.0),
        ]
    end
end
