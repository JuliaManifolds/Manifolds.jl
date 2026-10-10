using Manifolds, Random, Test

Test.@testset "Centered Matrices" begin

    Test.@testset "tangent vectors of any length" begin
        N = CenteredMatrices(3, 2)
        p = [1.0 2.0; -3.0 0.0; 2.0 -2.0]
        Test.@test is_vector(N, p, project(N, p, [100.0 -50.0; 30.0 20.0; -10.0 5.0]))
        Test.@test !is_vector(N, p, [1.0 1.0; -1.0 -1.0; 1.0e-6 0.0])
        Test.@test Weingarten!(N, similar(p), p, p, ones(3, 2)) == zero(p)
    end

    Test.@testset "points of any size" begin
        N = CenteredMatrices(3, 2)
        Test.@test is_point(N, project(N, [100.0 -50.0; 30.0 20.0; -10.0 5.0]))
        Test.@test !is_point(N, [1.0 1.0; -1.0 -1.0; 1.0e-6 0.0])
        Test.@test !is_point(N, [Inf 0.0; 0.0 0.0; 0.0 0.0])
    end

    M = CenteredMatrices(3, 2)

    p1 = [1.0 2.0; 4.0 5.0; -5.0 -7.0]
    p2 = [0 0; 1 -1; -1 1]
    p3 = [0.5 1; -1 -0.7; 0.5 -0.3]
    q1 = [1 2 3; 4 5 6; -5 -7 -9]    #wrong dimensions
    q2 = [-3 -im; 2 im; 1 0]         #complex
    q3 = [1.0 2; 3 4; 5 6]             #not centered
    c1 = [-3 / sqrt(2), 15 / sqrt(6), -3 / sqrt(2), 21 / sqrt(6)] # coordinates of p1

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                embed,
                get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                is_point, is_vector, is_flat,
                manifold_dimension,
                project, rand,
                repr, representation_size,
            ],
            :Points => [p1, p2, p3],
            :Vectors => [p1],
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [c1],
            :Rng => Random.Xoshiro(42),
            :EmbeddedPoints => [p1],
            :InvalidPoints => Matrix[q1, q2, q3], #To avoid implicit conversion to complex matrices
            :InvalidVectors => Matrix[q1, q2, q3],
        ),
        Dict(
            :IsPointErrors => [ManifoldDomainError, ManifoldDomainError, DomainError],
            :IsVectorErrors => [ManifoldDomainError, ManifoldDomainError, DomainError],
            (get_coordinates, DefaultOrthonormalBasis()) => c1,
            (get_vectors, DefaultOrthonormalBasis()) => :Orthonormal,
            :atols => Dict(get_vectors => 1.0e-15),
            is_flat => true,
            get_embedding => Euclidean(3, 2),
            manifold_dimension => 4,
            repr => "CenteredMatrices(3, 2, ℝ)",
            representation_size => (3, 2),
        )
    )
    Mf = CenteredMatrices(3, 2; parameter = :field)
    Manifolds.Test.test_manifold(
        Mf,
        Dict(:Functions => [repr, get_embedding]),
        Dict(
            repr => "CenteredMatrices(3, 2, ℝ; parameter=:field)",
            get_embedding => Euclidean(3, 2; parameter = :field),
        )
    )

    Mc = CenteredMatrices(3, 2, ℂ)
    p4 = [-3.0 -1.0im; 2.0 1.0im; 1.0 0.0]
    p5 = [1.0 1.0im; -1.0im 0.0; -1.0 + 1.0im -1.0im]
    p6 = [1.0im 0.0; -2.0im 1.0im; 1.0im -1.0im]
    q4 = [1.0im 0.0; -2.0im 1.0im; 1.0im 0.0] #complex and not centered
    c4 = [-5 / sqrt(2), -3 / sqrt(6), 0.0, 0.0, 0.0, 0.0, -sqrt(2), 0.0] # coordinates of p4
    # Complex case
    Manifolds.Test.test_manifold(
        Mc,
        Dict(
            :Functions => [
                embed,
                get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                is_point, is_vector, is_flat,
                manifold_dimension,
                project, rand,
                repr, representation_size,
            ],
            :Points => [p4, p5, p6],
            :Vectors => [p4],
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [c4],
            :Rng => Random.Xoshiro(42),
            :EmbeddedPoints => [p4],
            :InvalidPoints => Matrix[q1, q3, q4],
            :InvalidVectors => Matrix[q1, q3, q4],
        ),
        Dict(
            :IsPointErrors => [ManifoldDomainError, DomainError, DomainError],
            :IsVectorErrors => [ManifoldDomainError, DomainError, DomainError],
            (get_coordinates, DefaultOrthonormalBasis()) => c4,
            is_flat => true,
            get_embedding => Euclidean(3, 2; field = ℂ),
            manifold_dimension => 8,
            repr => "CenteredMatrices(3, 2, ℂ)",
            representation_size => (3, 2),
        )
    )

end
