using ManifoldDiff, Manifolds, Random, StaticArrays, Test

Test.@testset "The circle manifold" begin
    M = Circle()
    @test Manifolds.number_of_coordinates(M, DefaultOrthogonalBasis()) == 1
    p1 = π / 2
    p2 = -π / 2
    X1 = 1.0
    X2 = -1.0

    q1 = 9.0
    q2 = zeros(2, 2)
    Y1 = zeros(2, 2)

    expectations = Dict(
        manifold_dimension => 1,
        representation_size => (),
        repr => "Circle(ℝ)",
        manifold_volume => 2π,
    )
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                exp,
                get_basis, get_coordinates, get_vector,
                is_flat, is_point, is_vector,
                log,
                manifold_dimension, manifold_volume, mid_point,
                parallel_transport_direction, parallel_transport_to,
                repr, representation_size,
            ],
            :Bases => [DefaultOrthonormalBasis(), DiagonalizingOrthonormalBasis(X1)],
            :Coordinates => [[π / 2], [-π / 2]],
            :InvalidPoints => [q1, q2],
            :InvalidVectors => [Y1],
            :Mutating => false,
            :Points => [p1, p2],
            :Vectors => [X1, X2],
            :VectorTransportMethods => [ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport()],
        ),
        expectations,
    )
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                distance,
                exp,
                get_basis, get_coordinates, get_vector,
                inner, is_flat, is_point, is_vector,
                log,
                manifold_dimension, manifold_volume, mid_point,
                repr, representation_size,
            ],
            :Bases => [DefaultOrthonormalBasis(), DiagonalizingOrthonormalBasis([X1])],
            :Coordinates => [[π / 2], [-π / 2]],
            :InvalidPoints => fill.([q1]),
            :Points => fill.([p1, p2]),
            :Vectors => fill.([X1, X2]),
            :VectorTransportMethods => [ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport()],
        ),
        expectations,
    )

    Mc = Circle(ℂ)

    pc1 = 1.0im
    pc2 = 1.0
    Xc1 = 0.5
    Xc2 = 0.25

    qc1 = 1.0 + 1.0im
    Yc1 = 1.0im

    expectations = Dict(
        manifold_dimension => 1,
        representation_size => (),
        repr => "Circle(ℂ)",
        manifold_volume => 2π,
        get_embedding => Euclidean(; field = ℂ),
        :atols => Dict(parallel_transport_to => 1.0e-14),
    )
    Manifolds.Test.test_manifold(
        Mc,
        Dict(
            :Functions => [
                embed, exp,
                get_coordinates, get_embedding, get_vector,
                inner, injectivity_radius, is_flat, is_point, is_vector,
                manifold_dimension, manifold_volume, mid_point,
                project, parallel_transport_direction, parallel_transport_to,
                log,
                rand, repr, representation_size,
            ],
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [[π / 2], [-π / 2]],
            :EmbeddedPoints => [pc1],
            :EmbeddedVectors => [Xc1],
            :InvalidPoints => [qc1],
            :InvalidVectors => [Yc1],
            :Mutating => false,
            :Points => [pc1, pc2],
            :Vectors => [Xc1, Xc2],
        ),
        expectations
    )
    Manifolds.Test.test_manifold(
        Mc,
        Dict(
            :Functions => [
                exp,
                get_coordinates, get_vector,
                inner, injectivity_radius, is_flat, is_point, is_vector,
                manifold_dimension, manifold_volume, mid_point,
                parallel_transport_direction, parallel_transport_to, project,
                log,
                repr, representation_size,
            ],
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [[π / 2], [-π / 2]],
            :EmbeddedPoints => fill.([pc1]),
            :EmbeddedVectors => fill.([Xc1]),
            :InvalidPoints => fill.([qc1]),
            :InvalidVectors => fill.([Yc1]),
            :Points => fill.([pc1, pc2]),
            :Vectors => fill.([Xc1, Xc2]),
        ),
        expectations
    )

    Test.@testset "Edge cases" begin
        Test.@testset "Mean" begin
            M = Circle()
            @test mean(M, [-π / 2, 0.0, π]) ≈ -π / 2
            @test mean(M, [-π / 2, 0.0, π], [1.0, 1.0, 1.0]) == -π / 2
            Mc = Circle(ℂ)
            angles = map(pp -> exp(pp * im), [-π / 2, 0.0, π])
            @test mean(Mc, angles) ≈ exp(-π * im / 2)
            @test mean(Mc, angles, [1.0, 1.0, 1.0]) ≈ exp(-π * im / 2)

        end
        @testset "Mutating Rand for real Circle" begin
            M = Circle()
            p = fill(NaN)
            X = fill(NaN)
            rand!(M, p)
            @test is_point(M, p)
            rand!(M, X; vector_at = p)
            @test is_vector(M, p, X)

            rng = MersenneTwister()
            rand!(rng, M, p)
            @test is_point(M, p)
            rand!(rng, M, X; vector_at = p)
            @test is_vector(M, p, X)
        end
        @testset "Test sym_rem" begin
            M = Circle()
            p = 4.0 # not a point
            p = sym_rem(p) # modulo to a point
            @test is_point(M, p)
        end
        @testset "small and large distance tests" begin
            M = Circle(ℂ)
            p = -0.42681766710748265 + 0.9043377018818392im
            q = -0.42681766710748226 + 0.9043377018818393im
            @test isapprox(distance(Mc, p, q), 4.041272810440265e-16)
            @test isapprox(distance(Mc, p, -q), 3.1415926535897927; atol = eps())
        end
        @testset "Mixed array dimensions for exp and PT" begin
            # this is an issue on Julia 1.6 but not later releases
            M = Circle()
            p = fill(0.0)
            Manifolds.exp_fused!(M, p, p, [1.0], 2.0)
            @test p ≈ fill(2.0)
            parallel_transport_to!(M, p, p, [4.0], p)
            @test p ≈ fill(4.0)
        end
        Test.@testset "retract nonmutating defaults" begin
            M = Circle()
            p = π / 3
            X = 0.5
            d = -0.2
            q = retract(M, p, X)
            q2 = retract(M, p, X, ExponentialRetraction())
            @test q ≈ exp(M, p, X)
            @test q2 ≈ exp(M, p, X)
            @test vector_transport_direction(M, p, X, d, ParallelTransport()) == parallel_transport_to(M, p, X, d)
        end
        Test.@testset "ManifoldsDiff cases" begin
            M = Circle()
            @test ManifoldDiff.adjoint_Jacobi_field(
                M, 0.0, 1.0, 0.5, 2.0,
                ManifoldDiff.βdifferential_shortest_geodesic_startpoint,
            ) === 2.0
            @test ManifoldDiff.diagonalizing_projectors(M, 0.0, 2.0) == ((0.0, ManifoldDiff.ProjectorOntoVector(M, 0.0, SA[1.0])),)
            @test ManifoldDiff.jacobi_field(
                M, 0.0, 1.0, 0.5, 2.0,
                ManifoldDiff.βdifferential_shortest_geodesic_startpoint,
            ) === 2.0

            # volume
            @test manifold_volume(M) ≈ 2 * π
            @test volume_density(M, 0.0, 2.0) == 1.0
        end
        Test.@testset "Complex Circle log boundary case" begin
            Mc = Circle(ℂ)
            X = log(Mc, 1.0 + 0.0im, -1.0 + 0.0im)
            @test isapprox(X, π * 1.0im)
            X2 = log(Mc, fill(0 + 1.0im), fill(0.0 - 1.0im))
            @test isapprox(X2[], π)
            X3 = fill(0.0)
            log!(Mc, X3, fill(0 + 1.0im), fill(0.0 - 1.0im))
            @test isapprox(X3[], X2[])
        end
    end

    @testset "StaticArrays.jl and vector tests" begin
        Mc = Circle(ℂ)
        @test mid_point(Mc, Scalar(1.0 + 0.0im), Scalar(0.0 + 1.0im)) ≈ Scalar(sqrt(2) / 2 + sqrt(2) / 2 * 1im)
        X1 = get_vector(Mc, Scalar(1.0 + 0.0im), 2.0)
        @test X1 ≈ Scalar(2.0im)
        @test X1 isa Scalar{ComplexF64}

        M = Circle()
        @test get_vector(M, [1.0], [2.0]) == [2.0]
        @test get_vector(M, Scalar(1.0), SA[2.0]) isa SArray
    end
end
