using LinearAlgebra, Manifolds, Quaternions, Test

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
        )
    )

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
            :InvalidPoints => [pHF],
            :InvalidVectors => [Quaternion(1, 0, 0, 0)],
            :Mutating => false,
        ),
        Dict(
            default_vector_transport_method => ProjectionTransport(),
            get_embedding => Euclidean(1, 1; field = ℍ),
            is_flat => false,
            manifold_dimension => 3,
            repr => "UnitaryMatrices(1, ℍ)",
        )
    )


end
