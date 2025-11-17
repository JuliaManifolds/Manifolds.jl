s = joinpath(@__DIR__, "..", "ManifoldsTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))
using ManifoldsTestSuite

using LinearAlgebra, Manifolds, Test, StaticArrays, Quaternions, Random, Distributions, RecursiveArrayTools
using ManifoldDiff


@testset "The Spheres" begin

    M = Sphere(2)
    p = [1.0, 0.0, 0.0]
    q = [1 / sqrt(2), 1 / sqrt(2), 0.0]
    X = [0.0, π / 4, 0.0]
    Y = [0.0, 0.0, π / 4]
    V = [2.0, 0.0, 0.0]  # normal vector at p
    ξ = [0.0, π / 4, 0.0]  # covector at p

    @testset "Basics" begin
        @test is_default_metric(M, EuclideanMetric())
        @test !is_default_metric(M, AffineInvariantMetric())
        @test base_manifold(M) === M
        @test representation_size(M) == (3,)
        @test !is_flat(M)
        @test is_flat(Sphere(1))
        @test !is_point(M, [2.0, 0.0, 0.0])
        @test !is_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        @test_throws DomainError is_vector(
            M,
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0];
            error = :error,
        )
    end

    # TODO: test ProjectedOrthonormalBasis(:svd), DiagonalizingOrthonormalBasis

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => Manifolds.Test.all_functions(),
            :InverseRetractionMethods => [LogarithmicInverseRetraction(), ProjectionInverseRetraction()],
            :Points => [p, q], :Vectors => [X, Y],
            :EmbeddedPoints => [p],
            :EmbeddedVectors => [X],
            :Coordinates => [[π / 2, 0.0]],
            :Bases => [DefaultOrthonormalBasis()],
            :InvalidPoints => [2 * p],
            :InvalidVectors => [p],
            :NormalVectors => [V],
            :Covectors => [ξ],
            :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
            :VectorTransportMethods => [ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport()]
        ),
        # Expectations
        Dict(
            :atol => 1.0e-7,
            default_inverse_retraction_method => LogarithmicInverseRetraction(),
            default_retraction_method => StabilizedRetraction(),
            default_vector_transport_method => ParallelTransport(),
            distance => π / 4,
            exp => q,
            get_embedding => Euclidean(3),
            injectivity_radius => π,
            (injectivity_radius, ProjectionRetraction()) => π / 2,
            log => X, norm => π / 4,
            parallel_transport_to => parallel_transport_to(M, p, X, q),
            parallel_transport_direction => parallel_transport_to(M, p, X, q),
            manifold_dimension => 2,
            repr => "Sphere(2)",
            Weingarten => -X * (p'V), # or [ 0, -π/2, 0 ]
            :IsPointErrors => [DomainError]
        ),
    )

    @testset "Embedding test" begin
        p = [1.0, 0.0, 0.0]
        X = [0.0, 1.0, 1.0]
        @test embed(M, p) == p
        q = similar(p)
        embed!(M, q, p)
        @test q == p
        @test embed(M, p, X) == X
        Y = similar(X)
        embed!(M, Y, p, X)
        @test Y == X
    end

    @testset "Complex Sphere" begin
        M = Sphere(2, ℂ)

        p = [1.0, 1.0im, 1.0]
        q = project(M, p)
        Y = [2.0, 1.0im, 20.0]
        X = project(M, q, Y)

        Manifolds.Test.test_manifold(
            M,
            Dict(:Functions => [repr, get_embedding], :Points => [q], :Vectors => [X]),
            # Expectations
            Dict(
                repr => "Sphere(2, ℂ)",
                get_embedding => Euclidean(3; field = ℂ)
            ),
        )

        # testing that the random point is actually a complex number with non-zero imaginary part
        Random.seed!(42)
        r = rand(M)
        @test is_point(M, r)
        @test norm(imag.(r)) != 0
    end

    @testset "Quaternion Sphere" begin
        M = Sphere(2, ℍ)
        p = [Quaternion(1.0), Quaternion(0, 1.0, 0, 0), Quaternion(0.0, 0.0, -1.0, 0.0)]
        q = project(M, p)

        Y = [Quaternion(2.0), Quaternion(0, 1.0, 0, 0), Quaternion(0.0, 0.0, 20.0, 0.0)]
        X = project(M, q, Y)

        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Functions => [repr, get_embedding, project],
                :Points => [p],
                :EmbeddedPoints => [p],
                :Vectors => [X],
                :EmbeddedVectors => [X],
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14, repr => "Sphere(2, ℍ)",
                get_embedding => Euclidean(3; field = ℍ)
            ),
        )
    end

    Manifolds.Test.test_manifold(
        Sphere(2; parameter = :field),
        Dict(
            :Functions => [get_embedding, repr],
            :Points => [p],
        ),
        # Expectations
        Dict(
            repr => "Sphere(2; parameter=:field)",
            get_embedding => Euclidean(3; parameter = :field),
        ),
    )

    @testset "Array Sphere" begin
        M = ArraySphere(2, 2; field = ℝ)
        @test representation_size(M) == (2, 2)
        p = ones(2, 2)
        q = project(M, p)
        @test is_point(M, q)
        Y = [1.0 0.0; 0.0 1.1]
        X = project(M, q, Y)
        @test is_vector(M, q, X)

        Manifolds.Test.test_manifold(
            M,
            Dict(:Functions => [repr, get_embedding], :Points => [q], :Vectors => [X]),
            # Expectations
            Dict(repr => "ArraySphere(2, 2)", get_embedding => Euclidean(2, 2)),
        )
    end


    Manifolds.Test.test_manifold(
        ArraySphere(2, 2; field = ℂ),
        Dict(:Functions => [repr, get_embedding]),
        # Expectations
        Dict(repr => "ArraySphere(2, 2; field=ℂ)", get_embedding => Euclidean(2, 2; field = ℂ)),
    )

    Manifolds.Test.test_manifold(
        ArraySphere(2, 2; parameter = :field),
        Dict(:Functions => [repr]),
        # Expectations
        Dict(repr => "ArraySphere(2, 2; parameter=:field)"),
    )

    @testset "Specific Tests and boundary cases" begin
        M = Sphere(2)
        @testset "small and large distance tests" begin
            p = [-0.18337624444127734, 0.8345313166281056, 0.5195484910396462]
            q = [-0.18337624444127681, 0.8345313166281058, 0.5195484910396464]
            @test isapprox(distance(M, p, q), 5.828670879282073e-16)
            @test isapprox(distance(M, p, -q), 3.1415926535897927; atol = eps())
        end

        @testset "DefaultOrthonormalBasis edge cases" begin
            B = DefaultOrthonormalBasis(ℝ)
            n = manifold_dimension(M)
            p = [1; zeros(n)]
            for i in 1:n  # p'x ± 1
                vcoord = [j == i for j in 1:n]
                v = [0; vcoord]
                @test get_coordinates(M, p, v, B) ≈ vcoord
                @test get_vector(M, p, vcoord, B) ≈ v
                @test get_coordinates(M, -p, v, B) ≈ vcoord
                @test get_vector(M, -p, vcoord, B) ≈ v
            end
            p = [0; 1; zeros(n - 1)] # p'x = 0
            for i in 1:n
                vcoord = [j == i for j in 1:n]
                v = get_vector(M, p, vcoord, B)
                @test is_vector(M, p, v)
                @test get_coordinates(M, p, v, B) ≈ vcoord
            end
        end

        @testset "log edge case" begin
            n = manifold_dimension(M)
            x = normalize(randn(n + 1))
            v = log(M, x, -x)
            @test norm(v) ≈ π
            @test isapprox(dot(x, v), 0; atol = 1.0e-12)
            vexp = normalize(project(M, x, [1, zeros(n)...]))
            @test v ≈ π * vexp

            x = [1, zeros(n)...]
            v = log(M, x, -x)
            @test norm(v) ≈ π
            @test isapprox(dot(x, v), 0; atol = 1.0e-12)
            vexp = normalize(project(M, x, [0, 1, zeros(n - 1)...]))
            @test v ≈ π * vexp
        end

        @testset "StereographicAtlas" begin
            M = Sphere(2)
            A = Manifolds.StereographicAtlas()
            p = [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)]
            for k in [1, -1]
                p *= k
                i = Manifolds.get_chart_index(M, A, p)
                @test i === (p[1] < 0 ? :south : :north)
                a = get_parameters(M, A, i, p)
                q = get_point(M, A, i, a)
                @test isapprox(M, p, q)

                p2 = [1 / sqrt(2), 1 / sqrt(2), 0]; p3 = [0, 1 / sqrt(2), 1 / sqrt(2)]
                X2 = log(M, p, p2); X3 = log(M, p, p3)
                B = induced_basis(M, A, i, Manifolds.TangentSpaceType())

                X2B = get_coordinates(M, p, X2, B)
                X3B = get_coordinates(M, p, X3, B)

                @test inner(M, p, X2, X3) ≈ dot(X2B, local_metric(M, p, B) * X3B)

                X2back = get_vector(M, p, X2B, B)
                X3back = get_vector(M, p, X3B, B)
                @test isapprox(M, p, X2, X2back)
                @test isapprox(M, p, X3, X3back)

                @testset "transition_map" begin
                    other_chart = i === :south ? :north : :south
                    a_other = Manifolds.transition_map(M, A, i, other_chart, a)
                    @test isapprox(M, p, get_point(M, A, other_chart, a_other))

                    a_other2 = allocate(a_other)
                    Manifolds.transition_map!(M, a_other2, A, i, A, other_chart, a)
                    @test isapprox(M, p, get_point(M, A, other_chart, a_other2))

                    a_other2 = allocate(a_other)
                    Manifolds.transition_map!(M, a_other2, A, i, other_chart, a)
                    @test isapprox(M, p, get_point(M, A, other_chart, a_other2))
                end
            end
            # TODO: test RetractionAtlas somewhere
        end

        @testset "sectional curvature" begin
            M = Sphere(2; parameter = :field)
            K = Manifolds.sectional_curvature_matrix(
                M,
                [1.0, 0.0, 0.0],
                DefaultOrthonormalBasis(),
            )
            @test isapprox(K, [0 1; 1 0])
            @test isapprox(
                Manifolds.estimated_sectional_curvature_matrix(
                    M,
                    [1.0, 0.0, 0.0],
                    DefaultOrthonormalBasis(),
                ),
                [0.0 1.0; 1.0 0.0],
                atol = 0.15,
            )
        end

        @testset "Local Metric" begin
            p = [1.0, 0.0, 0.0]
            for M in [Sphere(2), Sphere(2; parameter = :field)]
                @test local_metric(M, p, DefaultOrthonormalBasis()) == Diagonal([1.0, 1.0])
            end
        end

        #= Add the following to the test suite as functions
            @testset "Weingarten" begin
                M = Sphere(2)
                p = [1.0, 0.0, 0.0]
                X = [0.0, 2.0, 0.0]
                V = [0.3, 0.0, 0.0]
                Y = -X * (p'V)
                @test Weingarten(M, p, X, V) == Y
            end
        =#

        @testset "Volume density" begin
            M = Sphere(2)
            p = [1.0, 0.0, 0.0]
            @test manifold_volume(Sphere(0)) ≈ 2
            @test manifold_volume(Sphere(1)) ≈ 2 * π
            @test manifold_volume(Sphere(2)) ≈ 4 * π
            @test manifold_volume(Sphere(3)) ≈ 2 * π * π
            @test volume_density(M, p, [0.0, 0.5, 0.5]) ≈ 0.9187253698655684
        end

        @testset "ManifoldDiff" begin
            # ManifoldDiff
            M = Sphere(2)
            p = [1.0, 0.0, 0.0]
            X = [0.0, 2.0, -1.0]
            dpX = ManifoldDiff.diagonalizing_projectors(M, p, X)
            @test dpX[1][1] == 0.0
            @test dpX[1][2].X == normalize(X)
            @test dpX[2][1] == 1.0
            @test dpX[2][2].X == normalize(X)
        end

        @testset "other metric" begin
            M = Sphere(2)
            p = [0, 1, 0]
            X = [1, 0, 0]
            Y = [1, 0, 4]
            Z = [0, 0, 1]
            @test riemann_tensor(M, p, X, Y, Z) == [4, 0, 0]
            @test sectional_curvature(M, p, X, Y) == 1.0
            @test sectional_curvature_max(M) == 1.0
            @test sectional_curvature_min(M) == 1.0
            M1 = Sphere(1)
            @test sectional_curvature(M1, p, X, Y) == 0.0
            @test sectional_curvature_max(M1) == 0.0
            @test sectional_curvature_min(M1) == 0.0
        end

        @testset "Distributions" begin
            sphere = Sphere(2)
            haar_measure = Manifolds.uniform_distribution(sphere)
            pts = rand(haar_measure, 5)
            @test all(p -> is_point(sphere, p), pts)

            usd_mvector = Manifolds.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
            @test isa(rand(usd_mvector), MVector)

            gtsd_mvector =
                Manifolds.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
            @test isa(rand(gtsd_mvector), MVector)
        end
    end
end
