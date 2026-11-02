using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Kendall's shape spaces" begin
    # an orthonormal basis of the centred landmark vectors of a triangle
    u = [1.0, -1.0, 0.0] / sqrt(2)
    v = [1.0, 1.0, -2.0] / sqrt(6)
    # an equilateral triangle and a horizontal unit tangent vector at it, i.e. f is orthogonal
    # to p and to the rotations of p, so the geodesic in direction f is horizontal
    p = [u'; v'] / sqrt(2)
    f = [v'; u'] / sqrt(2)
    Jp = [0.0 -1.0; 1.0 0.0] * p # the vertical direction, an infinitesimal rotation of p
    θ = π / 6
    q = cos(θ) * p + sin(θ) * f
    X = θ * f # log(M, p, q)
    r = [
        0.4385117672460505 -0.6877826444042382 0.24927087715818771
        -0.3830259932279294 0.35347460720654283 0.029551386021386548
    ]
    # row-wise constants are normal to the centred matrices
    shift = [1.0 1.0 1.0; -0.5 -0.5 -0.5]

    @testset "KendallsPreShapeSpace" begin
        @test_throws DomainError KendallsPreShapeSpace(2, 1)
        M = KendallsPreShapeSpace(2, 3)
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [[1.0, 2.0, 3.0]],
                :Covectors => [flat(M, p, X)],
                :EmbeddedPoints => [2 .* p .+ shift],
                :EmbeddedVectors => [X .+ 3 .* p .+ shift],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                    default_vector_transport_method, distance, embed, exp, flat, geodesic,
                    get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                    injectivity_radius, inner, inverse_retract, is_flat, is_point, is_vector, log,
                    manifold_dimension, manifold_volume, mid_point, norm,
                    parallel_transport_direction, parallel_transport_to, project, rand, repr,
                    representation_size, retract, sectional_curvature, sharp, shortest_geodesic,
                    vector_transport_direction, vector_transport_to, volume_density, Weingarten,
                    zero_vector,
                ],
                :InvalidPoints => [[1.0 0.0 1.0; 1.0 -1.0 0.0] / 2, 2 .* p],
                # orthogonal to p, but its rows do not have zero mean
                :InvalidVectors => [[1.0 1.0 1.0; 0.0 0.0 0.0], p],
                :InverseRetractionMethods => [
                    LogarithmicInverseRetraction(), ProjectionInverseRetraction(),
                ],
                :NormalVectors => [2 .* p .+ shift],
                :Points => [p, q, r],
                :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
                :SecondVector => Jp,
                :Vectors => [X, log(M, q, r), log(M, r, p)],
                :VectorTransportMethods => [ParallelTransport(), ProjectionTransport()],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => StabilizedRetraction(),
                default_vector_transport_method => ParallelTransport(),
                distance => θ,
                (embed, :Point) => p,
                (embed, :Vector) => X,
                exp => q,
                get_embedding => ArraySphere(2, 3),
                (get_vectors, DefaultOrthonormalBasis()) => :Orthonormal,
                injectivity_radius => π,
                (injectivity_radius, ProjectionRetraction()) => π / 2,
                inner => 0.0,
                is_flat => false,
                log => X,
                manifold_dimension => 3,
                # the pre-shape space of triangles is the 3-sphere
                manifold_volume => 2 * π^2,
                mid_point => cos(θ / 2) * p + sin(θ / 2) * f,
                norm => θ,
                # the velocity of the geodesic at q
                parallel_transport_direction => θ * (cos(θ) * f - sin(θ) * p),
                parallel_transport_to => θ * (cos(θ) * f - sin(θ) * p),
                (project, :Point) => p,
                (project, :Vector) => X,
                (retract, ExponentialRetraction()) => q,
                repr => "KendallsPreShapeSpace(2, 3)",
                representation_size => (2, 3),
                sectional_curvature => 1.0,
                sectional_curvature_max => 1.0,
                sectional_curvature_min => 1.0,
                sharp => X,
                (vector_transport_to, ParallelTransport()) => θ * (cos(θ) * f - sin(θ) * p),
                (vector_transport_to, ProjectionTransport()) => X - dot(q, X) * q,
                volume_density => (sin(θ) / θ)^2,
                Weingarten => -2 * X,
                :IsPointErrors => [DomainError, ManifoldDomainError],
                :IsVectorErrors => [DomainError, ManifoldDomainError],
                :IsVectorBasepointError => ManifoldDomainError,
            ),
        )

        @testset "random points and vectors use the element type of the array" begin
            pB = rand!(MersenneTwister(42), M, zeros(BigFloat, 2, 3))
            @test pB == project(M, randn(MersenneTwister(42), BigFloat, 2, 3))
            @test is_point(M, pB; error = :error)
            XB = rand!(MersenneTwister(7), M, zeros(BigFloat, 2, 3); vector_at = pB)
            @test XB == project(M, pB, randn(MersenneTwister(7), BigFloat, 2, 3))
        end

        @testset "coordinates in an orthonormal basis" begin
            B = DefaultOrthonormalBasis()
            M3 = KendallsPreShapeSpace(3, 4)
            p3 = project(M3, [1.0 2.0 -0.5 0.3; -1.0 0.4 0.8 2.0; 0.2 -0.7 1.5 -1.1])
            X3 = project(M3, p3, [0.5 -0.3 0.2 0.1; 0.0 0.7 -0.4 0.6; -0.8 0.1 0.9 -0.2])
            Y3 = project(M3, p3, [0.2 0.6 -0.1 -0.9; 0.4 -0.5 0.3 0.8; 0.1 0.2 -0.7 0.5])
            c = get_coordinates(M3, p3, X3, B)
            @test length(c) == manifold_dimension(M3)
            @test isapprox(M3, p3, get_vector(M3, p3, c, B), X3)
            @test is_vector(M3, p3, get_vector(M3, p3, c, B))
            @test dot(c, get_coordinates(M3, p3, Y3, B)) ≈ inner(M3, p3, X3, Y3)
        end

        Manifolds.Test.test_manifold(
            KendallsPreShapeSpace(2, 3; parameter = :field),
            Dict(:Functions => [get_embedding, is_point, repr], :Points => [p]),
            # Expectations
            Dict(
                get_embedding => ArraySphere(2, 3; parameter = :field),
                repr => "KendallsPreShapeSpace(2, 3; parameter=:field)",
            ),
        )
    end

    @testset "KendallsShapeSpace" begin
        @test_throws DomainError KendallsShapeSpace(0, 3)
        @test_throws DomainError KendallsShapeSpace(2, 1)
        M = KendallsShapeSpace(2, 3)
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Covectors => [flat(M, p, X)],
                :EmbeddedPoints => [2 .* p .+ shift],
                :EmbeddedVectors => [X .+ 3 .* p .+ Jp .+ shift],
                :Functions => [
                    copy, copyto!, distance, embed, exp, flat, geodesic, get_embedding, inner,
                    is_flat, is_point, is_vector, log, manifold_dimension, mid_point, norm,
                    project, rand, repr, representation_size, sharp, shortest_geodesic,
                    zero_vector,
                ],
                :InvalidPoints => [[1.0 0.0 1.0; 1.0 -1.0 0.0]],
                :InvalidVectors => [[1.0 0.0 1.0; 1.0 -1.0 0.0]],
                :Points => [p, q, r],
                # the vertical part does not change the inner product
                :SecondVector => f + Jp,
                :Vectors => [X, log(M, q, r), log(M, r, p)],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
                distance => θ,
                (embed, :Point) => p,
                (embed, :Vector) => X,
                exp => q,
                get_embedding => KendallsPreShapeSpace(2, 3),
                inner => θ,
                is_flat => false,
                log => X,
                manifold_dimension => 2,
                mid_point => cos(θ / 2) * p + sin(θ / 2) * f,
                norm => θ,
                (project, :Point) => p,
                (project, :Vector) => X,
                repr => "KendallsShapeSpace(2, 3)",
                representation_size => (2, 3),
                sharp => X,
                :IsPointErrors => [ManifoldDomainError],
                :IsVectorErrors => [ManifoldDomainError],
            ),
        )

        @testset "Basics" begin
            @test get_total_space(M) == KendallsPreShapeSpace(2, 3)
            @test metric(M) == EuclideanMetric()
            Yamb = [0.3 -0.2 0.5; 0.1 0.4 -0.6]
            @test riemannian_gradient(M, r, Yamb) == project(M, r, Yamb)
            @test manifold_dimension(KendallsShapeSpace(3, 2)) == 0
        end

        @testset "tangent vector components" begin
            X1 = [
                0.6090792159558263 -0.02523987621672985 -0.5838393397390964
                0.4317628895706799 0.12108361184633629 -0.5528465014170161
            ]
            X1h = [
                0.5218590427922166 0.05525104866717821 -0.5771100914593948
                0.3319078589730016 0.2777009756923593 -0.6096088346653609
            ]
            X1v = [
                0.08722017316360964 -0.08049092488390806 -0.006729248279701561
                0.09985503059767825 -0.156617363846023 0.05676233324834479
            ]
            @test horizontal_component(M, r, X1) ≈ X1h
            @test vertical_component(M, r, X1) ≈ X1v
            Y = similar(X1)
            vertical_component!(M, Y, r, X1)
            @test Y ≈ X1v
            @test norm(M, r, X1v) < 1.0e-16
            @test abs(norm(M, r, X1) - norm(M, r, X1h)) < 1.0e-16
            @test distance(M, r, exp(M, r, X1)) ≈ norm(M, r, X1)
        end

        @testset "project, rand and the projected basis give horizontal vectors" begin
            Y = project(M, r, [1.0 0.0 -1.0; 0.0 1.0 -1.0])
            @test norm(M, r, Y) ≈ norm(get_embedding(M), r, Y)
            Z = rand(MersenneTwister(44), M; vector_at = r)
            @test norm(M, r, Z) ≈ norm(get_embedding(M), r, Z)
            V = get_vectors(M, r, get_basis(M, r, ProjectedOrthonormalBasis(:svd)))
            @test [inner(M, r, v, w) for v in V, w in V] ≈ I
        end

        @testset "collinear landmarks and a triangle in space" begin
            pc = [1.0 -1.0 0.0; 0.0 0.0 0.0] ./ sqrt(2)
            Xc = [0.0 0.0 0.0; 1.0 -0.5 -0.5]
            @test horizontal_component(M, pc, Xc) ≈ [0.0 0.0 0.0; 0.25 0.25 -0.5]
            @test norm(M, pc, Xc) ≈ sqrt(0.375)
            @test distance(M, pc, exp(M, pc, Xc)) ≈ sqrt(0.375)
            qc = [0.5 -0.5 0.0; 0.5 -0.5 0.0]
            Xq = [-0.375 0.375 0.0; 0.625 -0.125 -0.5]
            @test horizontal_component(M, qc, Xq) ≈ [0.0 0.0 0.0; 0.25 0.25 -0.5]
            M3 = KendallsShapeSpace(3, 3)
            p3 = [0.5 -0.5 0.0; 0.0 0.5 -0.5; 0.0 0.0 0.0]
            X3 = [0.0 0.0 0.0; 1.0 -0.5 -0.5; 0.0 0.0 0.0]
            Y3 = [0.0 0.375 -0.375; 0.625 -0.125 -0.5; 0.0 0.0 0.0]
            @test horizontal_component(M3, p3, X3) ≈ Y3
        end

        Manifolds.Test.test_manifold(
            KendallsShapeSpace(2, 3; parameter = :field),
            Dict(:Functions => [get_embedding, is_point, repr], :Points => [p]),
            # Expectations
            Dict(
                get_embedding => KendallsPreShapeSpace(2, 3; parameter = :field),
                repr => "KendallsShapeSpace(2, 3; parameter=:field)",
            ),
        )
        @test get_total_space(KendallsShapeSpace(2, 3; parameter = :field)) ===
            KendallsPreShapeSpace(2, 3; parameter = :field)
    end
end
