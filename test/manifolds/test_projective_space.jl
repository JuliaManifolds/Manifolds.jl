using Distributions, LinearAlgebra, Manifolds, ManifoldsBase, Quaternions, Random
using RecursiveArrayTools, StaticArrays, Test

@testset "The projective spaces" begin
    @test_throws DomainError ProjectiveSpace(-1)
    @test_throws DomainError ArrayProjectiveSpace(0, 2)

    retraction_methods = [PolarRetraction(), ProjectionRetraction(), QRRetraction()]
    inverse_retraction_methods = [
        PolarInverseRetraction(), ProjectionInverseRetraction(), QRInverseRetraction(),
    ]
    vector_transport_methods = [
        ParallelTransport(), PoleLadderTransport(), ProjectionTransport(),
        SchildsLadderTransport(),
    ]
    functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, embed, exp, flat, geodesic, get_basis,
        get_coordinates, get_embedding, get_vector, get_vectors, injectivity_radius, inner,
        inverse_retract, is_flat, is_point, is_vector, log, manifold_dimension, mid_point,
        norm, parallel_transport_direction, parallel_transport_to, project, rand, repr,
        representation_size, retract, sharp, shortest_geodesic, vector_transport_direction,
        vector_transport_to, zero_vector,
    ]

    @testset "Real" begin
        M = ProjectiveSpace(2)
        p1 = [1.0, 0.0, 0.0]
        X = [0.0, 1.0, 0.0]
        X2 = [0.0, 1.0, -1.0]
        p2 = exp(M, p1, X)
        p3 = exp(M, p1, X2)
        # coordinates in the projected and diagonalizing bases require a cached basis
        bases = [
            DefaultOrthonormalBasis(), get_basis(M, p1, ProjectedOrthonormalBasis(:svd)),
            get_basis(M, p1, DiagonalizingOrthonormalBasis([0.0, 1.0, 2.0])),
        ]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => bases,
                :Coordinates => [get_coordinates(M, p1, X, B) for B in bases],
                :Covectors => [Manifolds.RieszRepresenterCotangentVector(M, p1, X)],
                :EmbeddedPoints => [2 * p1],
                :EmbeddedVectors => [[1.0, 1.0, 1.0]],
                :Functions => functions,
                :InvalidPoints => [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                :InvalidVectors => [[1.0, 0.0, 0.0]],
                :InverseRetractionMethods => inverse_retraction_methods,
                :Points => [p1, p2, p3],
                :RetractionMethods => retraction_methods,
                :SecondVector => X2,
                :Vectors => [X, project(M, p2, X2), project(M, p3, X)],
                :VectorTransportMethods => vector_transport_methods,
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                :atols => Dict(geodesic => 1.0e-11, shortest_geodesic => 1.0e-11),
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ParallelTransport(),
                distance => 1.0,
                exp => [cos(1), sin(1), 0.0],
                (get_coordinates, bases[1]) => [1.0, 0.0],
                (get_vectors, bases[1]) => :Orthonormal,
                (get_vectors, bases[2]) => :Orthonormal,
                (get_vectors, bases[3]) => :Orthonormal,
                get_embedding => Euclidean(3),
                injectivity_radius => π / 2,
                (injectivity_radius, PolarRetraction()) => π / 2,
                (injectivity_radius, ProjectionRetraction()) => π / 2,
                (injectivity_radius, QRRetraction()) => π / 2,
                inner => 1.0,
                is_flat => false,
                log => X,
                manifold_dimension => 2,
                mid_point => [cos(0.5), sin(0.5), 0.0],
                norm => 1.0,
                (project, :Point) => p1,
                (project, :Vector) => [0.0, 1.0, 1.0],
                (retract, ProjectionRetraction()) => [1.0, 1.0, 0.0] / sqrt(2),
                repr => "ProjectiveSpace(2, ℝ)",
                representation_size => (3,),
                sharp => X,
                :IsPointErrors => [ManifoldDomainError, DomainError],
                :IsVectorErrors => [DomainError],
            ),
        )

        @testset "Basics" begin
            @test metric(M) == EuclideanMetric()
            @test riemannian_gradient(M, p1, [1.0, 1.0, 1.0]) == [0.0, 1.0, 1.0]
            @test is_flat(ProjectiveSpace(1))
            @test !is_vector(M, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            # tangent vectors of any length are accepted, a clearly normal part is not
            p = [1.0, 2.0, 3.0] / sqrt(14.0)
            @test is_vector(M, p, project(M, p, [100.0, -50.0, 30.0]))
            @test !is_vector(M, p, project(M, p, [1.0, -0.5, 0.3]) .+ 1.0e-6 .* p)
        end

        @testset "equivalence" begin
            @test isapprox(M, p1, -p1)
            @test isapprox(M, p1, exp(M, p1, π * X))
            @test log(M, p1, -p1) ≈ zero(X)
            @test isapprox(M, -p1, vector_transport_to(M, p1, X, -p1), -X)
        end

        @testset "Distributions" begin
            pd = Manifolds.uniform_distribution(M, p1)
            @test is_point(M, rand(pd))
            Xd = Manifolds.normal_tvector_distribution(M, p1, 1.0)
            @test is_vector(M, p1, rand(Xd))
            upd_mvector = Manifolds.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
            @test isa(rand(upd_mvector), MVector)
            gtpd_mvector =
                Manifolds.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
            @test isa(rand(gtpd_mvector), MVector)
        end
    end

    @testset "Complex" begin
        M = ProjectiveSpace(2, ℂ)
        p1 = [0.5 + 0.5im, 0.5 + 0.5im, 0]
        X = [0.0, 0.0, 1.0 - im]
        # the first two points are not multiplied by a phase, since transports along
        # different routes then end up in different representatives
        p2 = exp(M, p1, X)
        Y = [0.5, -0.5, 0.5im]
        p3 = (sqrt(0.5) - sqrt(0.5) * im) * exp(M, p1, Y)
        bases = [DefaultOrthonormalBasis(ℂ)]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => bases,
                :Coordinates => [get_coordinates(M, p1, X, bases[1])],
                :Covectors => [Manifolds.RieszRepresenterCotangentVector(M, p1, X)],
                :EmbeddedPoints => [2 * p1],
                :EmbeddedVectors => [ComplexF64[1.0, 1.0im, 1.0]],
                :Functions => functions,
                :InvalidPoints => [[1.0 + 0im, 0.0, 0.0, 0.0], [1.0, im, 0.0]],
                :InvalidVectors => [[1.0 + 0im, 0.0, 0.0], [-0.5im, 0.0, 0.0]],
                :InverseRetractionMethods => inverse_retraction_methods,
                :Points => [p1, p2, p3],
                :RetractionMethods => retraction_methods,
                :SecondVector => project(M, p1, Y),
                :Vectors => [X, project(M, p2, Y), project(M, p3, X)],
                :VectorTransportMethods => vector_transport_methods,
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                :atols => Dict(geodesic => 1.0e-10, shortest_geodesic => 1.0e-10),
                distance => sqrt(2),
                exp => p2,
                (get_vectors, bases[1]) => :Orthonormal,
                get_embedding => Euclidean(3; field = ℂ),
                injectivity_radius => π / 2,
                is_flat => false,
                log => X,
                manifold_dimension => 4,
                norm => sqrt(2),
                (number_eltype, rand) => ComplexF64,
                (project, :Point) => p1,
                repr => "ProjectiveSpace(2, ℂ)",
                representation_size => (3,),
                sharp => X,
                :IsPointErrors => [ManifoldDomainError, DomainError],
                :IsVectorErrors => [DomainError, DomainError],
            ),
        )

        @testset "Basics" begin
            @test Manifolds.allocation_promotion_function(M, exp!, (1,)) == complex
            @test !is_vector(M, [1.0 + 0im, 0.0, 0.0, 0.0], [0.0 + 0im, 1.0, 0.0])
        end

        @testset "equivalence" begin
            x = [1.0 + 0im, 0.0, 0.0]
            v = [0.0, im, 0.0]
            s = sqrt(0.5) - sqrt(0.5) * im
            @test isapprox(M, x, s * x)
            @test isapprox(M, x, exp(M, x, π * v))
            @test log(M, x, s * x) ≈ zero(v)
            @test isapprox(M, s * x, vector_transport_to(M, x, v, s * x), s * v)
        end
    end

    @testset "Right quaternion" begin
        M = ProjectiveSpace(2, ℍ)
        p1 = [Quaternion(0.5, 0, 0, 0.5), Quaternion(0, 0, 0.5, 0.5), Quaternion(0)]
        X = [Quaternion(0), Quaternion(0), Quaternion(0.0, -0.5, -0.5, 0.0)]
        # the first two points are not multiplied by a phase, since transports along
        # different routes then end up in different representatives
        p2 = exp(M, p1, X)
        Y = project(M, p1, [Quaternion(0.25, -0.25, 0.25, 0.25), Quaternion(0.25, 0.25, -0.25, -0.25), Quaternion(1)])
        p3 = exp(M, p1, Y) * Quaternion(0.5, -0.5, 0.5, -0.5) # equivalence by right multiplication
        bases = [DefaultOrthonormalBasis(ℍ)]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => bases,
                :Coordinates => [get_coordinates(M, p1, X, bases[1])],
                :Covectors => [Manifolds.RieszRepresenterCotangentVector(M, p1, X)],
                :EmbeddedPoints => [2 * p1],
                :EmbeddedVectors => [quat([1.0, 0.5, 1.0])],
                :Functions => functions,
                :InvalidPoints => [quat([1.0, 0.0, 0.0, 0.0]), [1.0, quat(0, 1, 0, 0), 0.0]],
                :InvalidVectors => [Quaternion[1.0, 0.0, 0.0], quat([-0.5, 0.0, 0.0])],
                :InverseRetractionMethods => inverse_retraction_methods,
                :Points => [p1, p2, p3],
                :RetractionMethods => retraction_methods,
                :SecondVector => project(M, p1, Y),
                :Vectors => [X, project(M, p2, Y), project(M, p3, X)],
                :VectorTransportMethods => vector_transport_methods,
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                :atols => Dict(geodesic => 1.0e-10, shortest_geodesic => 1.0e-10),
                distance => sqrt(0.5),
                exp => p2,
                (get_vectors, bases[1]) => :Orthonormal,
                injectivity_radius => π / 2,
                is_flat => false,
                log => X,
                manifold_dimension => 8,
                norm => sqrt(0.5),
                (project, :Point) => p1,
                repr => "ProjectiveSpace(2, ℍ)",
                representation_size => (3,),
                sharp => X,
                :IsPointErrors => [ManifoldDomainError, DomainError],
                :IsVectorErrors => [DomainError, DomainError],
            ),
        )

        @testset "Basics" begin
            @test !is_vector(M, quat([1.0, 0.0, 0.0, 0.0]), quat([0.0, 1.0, 0.0]))
            @test !is_vector(M, quat([1.0, 0.0, 0.0]), [quat(0, -0.5, 0, 0), 0.0, 0.0])
        end

        @testset "equivalence" begin
            x = Quaternion[1.0, 0.0, 0.0]
            v = [0.0, quat(0, 1, 0, 0), 0.0]
            s = Quaternion(0.5, -0.5, 0.5, -0.5)
            @test isapprox(M, x, x * s)
            @test isapprox(M, x, exp(M, x, π * v))
            @test log(M, x, x * s) ≈ zero(v)
            @test isapprox(M, x * s, vector_transport_to(M, x, v, x * s), v * s)
        end
    end

    @testset "ArrayProjectiveSpace over $𝔽" for (𝔽, T, d) in
        [(ℝ, Float64, 3), (ℂ, ComplexF64, 6), (ℍ, Quaternion{Float64}, 12)]
        M = ArrayProjectiveSpace(2, 2; field = 𝔽)
        p1 = T[1.0 0.0; 0.0 0.0]
        p2 = T[1.0 1.0; 1.0 1.0] / 2
        X = T[0.0 1.0; 0.0 0.0]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :EmbeddedPoints => [2 * p1],
                :EmbeddedVectors => [T[1.0 1.0; 0.0 1.1]],
                :Functions => [
                    copy, copyto!, get_embedding, is_flat, is_point, is_vector,
                    manifold_dimension, project, rand, repr, representation_size,
                    zero_vector,
                ],
                :Points => [p1, p2],
                :Vectors => [X, project(M, p2, X)],
                :Seed => 4711,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-14,
                get_embedding => Euclidean(2, 2; field = 𝔽),
                is_flat => false,
                manifold_dimension => d,
                (project, :Point) => p1,
                (project, :Vector) => T[0.0 1.0; 0.0 1.1],
                repr => "ArrayProjectiveSpace(2, 2; field=$𝔽)",
                representation_size => (2, 2),
            ),
        )
        @test typeof(get_embedding(M)) === Euclidean{𝔽, ManifoldsBase.TypeParameter{Tuple{2, 2}}}
    end

    @testset "small distance tests" begin
        @testset for fT in (Float32, Float64), T in (fT, Complex{fT}, Quaternion{fT})
            𝔽 = T <: Complex ? ℂ : (T <: Quaternion ? ℍ : ℝ)
            M = ProjectiveSpace(2, 𝔽)
            rT = real(T)
            atol = rtol = sqrt(eps(rT))
            @testset for t in (zero(rT), eps(rT)^(1 // 4) / 8, eps(rT)^(1 // 4)),
                    λ in (one(T), (T <: Real ? -one(T) : sign(randn(T))))

                p = project(M, randn(T, representation_size(M)))
                @test is_point(M, p)
                X = project(M, p, randn(T, representation_size(M)))
                X ./= norm(M, p, X)
                project!(M, X, p, X)
                @test distance(M, p, exp(M, p, t * X) * λ) ≈ t atol = atol rtol = rtol
                @test distance(M, p, exp(M, p, (π - t) * X) * λ) ≈ t atol = atol rtol = rtol
            end
        end
    end

    @testset "Volume" begin
        @test manifold_volume(ProjectiveSpace(0)) ≈ 1
        @test manifold_volume(ProjectiveSpace(1)) ≈ π
        @test manifold_volume(ProjectiveSpace(2)) ≈ 2 * π
        @test manifold_volume(ProjectiveSpace(3)) ≈ π * π
    end

    @testset "field parameter" begin
        M = ProjectiveSpace(2; parameter = :field)
        @test typeof(get_embedding(M)) === Euclidean{ℝ, Tuple{Int}}
        @test typeof(get_embedding(ArrayProjectiveSpace(2, 3; parameter = :field))) ===
            Euclidean{ℝ, Tuple{Int, Int}}
        @test repr(M) == "ProjectiveSpace(2, ℝ; parameter=:field)"
        @test repr(ArrayProjectiveSpace(2, 3; parameter = :field)) ==
            "ArrayProjectiveSpace(2, 3; field=ℝ, parameter=:field)"
    end
end
