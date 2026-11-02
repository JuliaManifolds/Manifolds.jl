using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The essential manifold" begin
    Id = Matrix(1.0I, 3, 3)
    # generators of the rotations about the x, y and z axes
    Gx = [0.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 1.0 0.0]
    Gy = [0.0 0.0 1.0; 0.0 0.0 0.0; -1.0 0.0 0.0]
    Gz = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    Rx(t) = exp(t * Gx)
    Ry(t) = exp(t * Gy)
    Rz(t) = exp(t * Gz)
    a, b, c = π / 6, π / 4, π / 5
    # tangent vectors at (I, I) are horizontal iff their z components cancel,
    # so the turns about the x and y axes form a horizontal geodesic
    p = [Id, Id]
    q = [Rx(a), Ry(b)]
    r = [Ry(c), Rx(c)]
    X = [a * Gx, b * Gy] # log(M, p, q)
    Y = [Gx + Gz, -Gz]
    # Invalids
    nr = [1.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 -1.0 0.0] # a reflection
    B = DefaultOrthonormalBasis()

    @testset "$(name)" for (name, is_signed) in [("Signed", true), ("Unsigned", false)]
        M = EssentialManifold(is_signed)
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [B],
                :Coordinates => [[1.0, 0.0, 0.0, 0.0, 0.0]],
                :Covectors => [flat(M, p, X)],
                :EmbeddedPoints => [[2 * Id, 2 * Id]],
                # X with added symmetric and vertical parts
                :EmbeddedVectors => [[X[1] + Gz + [1.0 2.0 0.0; 2.0 0.0 0.0; 0.0 0.0 0.0], X[2] + Gz]],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                    default_vector_transport_method, distance, exp, flat, geodesic, get_basis,
                    get_coordinates, get_vector, get_vectors, injectivity_radius, inner,
                    inverse_retract, is_flat, is_point, is_vector, log, manifold_dimension,
                    mid_point, norm, parallel_transport_direction, parallel_transport_to,
                    project, rand, repr, retract, sharp, shortest_geodesic,
                    vector_transport_direction, vector_transport_to, zero_vector,
                ],
                :InvalidPoints => Any[Rx(a), [Rx(a), Ry(b), Rz(c)], [Rx(a), nr], [nr, nr]],
                :InvalidVectors => Any[zeros(3, 3), q],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [p, q, r],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => Y,
                :Vectors => [X, log(M, q, r), log(M, r, p)],
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ParallelTransport(),
                # the norm on the rotations is √2 times the angle
                distance => sqrt(2 * (a^2 + b^2)),
                exp => q,
                # X is orthogonal to the vertical direction, which the basis maps to the last coordinate
                (get_coordinates, B) => [sqrt(2) * a, 0.0, 0.0, 0.0, sqrt(2) * b],
                (get_vector, [1.0, 0.0, 0.0, 0.0, 0.0], B) => [Gx / sqrt(2), zeros(3, 3)],
                (get_vectors, B) => :Orthonormal,
                inner => 2 * a,
                is_flat => false,
                log => X,
                manifold_dimension => 5,
                mid_point => [Rx(a / 2), Ry(b / 2)],
                norm => sqrt(2 * (a^2 + b^2)),
                # the turns commute with the geodesic, so X is not changed
                parallel_transport_direction => X,
                parallel_transport_to => X,
                (project, :Point) => p,
                (project, :Vector) => X,
                (retract, ExponentialRetraction()) => q,
                repr => "EssentialManifold($(is_signed))",
                sharp => X,
                (vector_transport_to, ParallelTransport()) => X,
                :IsPointErrors => [
                    DomainError, DomainError, ComponentManifoldError, CompositeManifoldError,
                ],
                :IsVectorErrors => [DomainError, CompositeManifoldError],
                :IsVectorBasepointError => DomainError,
            ),
        )
    end

    @testset "Basics" begin
        M = EssentialManifold()
        @test M.manifold == Rotations(3)
        @test is_vector(M, q, project(M, q, log(M, q, r)))
        @test is_vector(M, q, project(M, q, log(M, r, q)))
    end

    @testset "the signed essential distance where a rotation angle reaches zero" begin
        M = EssentialManifold(true)
        # a half turn about z is shared by both cameras, one about x is not
        @test distance(M, p, [Id, Rz(π)]) ≈ π
        @test distance(M, p, [Rx(π), Id]) ≈ sqrt(2) * π
        @test distance(M, p, [Id, Rx(π)]) ≈ sqrt(2) * π
    end

    @testset "the unsigned essential distance is the same for every representative" begin
        M = EssentialManifold(false)
        # turns by 1 about the x axis and by 2 about the y axis, at distance sqrt(2 * (1^2 + 2^2))
        q = [Rx(1.0), Ry(2.0)]
        for h in [[Id, Id], [Rx(π), Rx(π)], [Id, Rz(π)], [Rx(π), Ry(π)]]
            @test distance(M, p, [h[1] * q[1], h[2] * q[2]]) ≈ sqrt(10)
        end
    end

    @testset "coordinates in a cached basis" begin
        M = EssentialManifold()
        p = rand(MersenneTwister(42), M)
        X = project(M, p, rand(MersenneTwister(44), M; vector_at = p))
        c = get_coordinates(M, p, X, B)
        @test length(c) == manifold_dimension(M)
        @test norm(c) ≈ norm(M, p, X)
        Bc = get_basis(M, p, B)
        @test get_coordinates(M, p, X, Bc) ≈ c
        @test get_coordinates!(M, similar(c), p, X, Bc) ≈ c
        @test isapprox(M, p, get_vector(M, p, c, Bc), X)
    end

    @testset "random tangent vectors on the essential manifold are horizontal" begin
        M = EssentialManifold(true)
        N = PowerManifold(Rotations(3), NestedPowerRepresentation(), 2)
        p = rand(MersenneTwister(5), M)
        X = rand(MersenneTwister(5), M; vector_at = p)
        @test distance(M, p, exp(M, p, X)) ≈ norm(M, p, X)
        # the same draw on the pair of rotations differs from X only by a vertical vector
        X0 = rand(MersenneTwister(5), N; vector_at = p)
        @test get_coordinates(M, p, X, B) ≈ get_coordinates(M, p, X0, B)
        Random.seed!(5)
        Y = rand(M; vector_at = p)
        @test norm(get_coordinates(M, p, Y, B)) ≈ norm(M, p, Y)
    end
end
