using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Veronese Manifold" begin
    @test Veronese(4, 3) == Veronese(4, 3; parameter = :type)
    @test typeof(Veronese(4, 3)) !== typeof(Veronese(5, 3))
    @test typeof(Veronese(4, 3; parameter = :field)) ===
        typeof(Veronese(5, 3; parameter = :field))
    @test_throws ArgumentError Veronese(0, 2)
    @test_throws ArgumentError Veronese(3, 0)
    @test_throws ArgumentError Veronese(3, 2; parameter = :invalid)

    n = 4

    x = normalize([1.0, 2.0, -1.0, 0.5])
    u = normalize([x[2], -x[1], 0.0, 0.0])
    v_raw = [0.5, -1.0, 2.0, 1.0]
    v = v_raw - dot(x, v_raw) * x

    z = normalize([1.0, -0.5, 0.25, 2.0])
    w_raw = [-0.5, 1.0, 0.5, 0.25]
    w = w_raw - dot(z, w_raw) * z

    for parameter in (:type, :field), d in 1:3
        M = Veronese(n, d; parameter)

        p = VeronesePoint([-0.7], x)
        q = VeronesePoint([iseven(d) ? -0.9 : 0.9], z)
        X = VeroneseTangentVector([0.2], u)
        Y = VeroneseTangentVector([-0.3], v)
        Z = VeroneseTangentVector([0.1], w)

        invalid_points = [
            VeronesePoint([1.0, 2.0], x),
            VeronesePoint([0.0], x),
            VeronesePoint([Inf], x),
            VeronesePoint([1.0], x[1:3]),
            VeronesePoint([1.0], 2 .* x),
        ]
        invalid_vectors = [
            VeroneseTangentVector([0.2, 0.1], u), VeroneseTangentVector([0.2], x),
        ]

        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Aliased => false,
                :Functions => [
                    default_inverse_retraction_method,
                    default_retraction_method,
                    default_vector_transport_method,
                    distance,
                    embed,
                    get_coordinates,
                    get_embedding,
                    get_vector,
                    inner,
                    is_point,
                    is_vector,
                    log,
                    manifold_dimension,
                    rand,
                    repr,
                ],
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [[0.2; zeros(n - 1)]],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :Points => [p, q],
                :RetractionMethods => [ExponentialRetraction()],
                :Rng => Random.Xoshiro(42),
                :SecondVector => Y,
                :TestVectorArithmetic => true,
                :VectorTransportMethods => [ProjectionTransport()],
                :Vectors => [X, Z],
            ),
            Dict(
                :atol => 1.0e-12,
                :IsPointErrors => fill(DomainError, length(invalid_points)),
                :IsVectorErrors => fill(DomainError, length(invalid_vectors)),
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ProjectionTransport(),
                get_embedding => Euclidean(n^d; parameter),
                manifold_dimension => n,
                repr => "Veronese($n, $d$(parameter === :field ? "; parameter=:field" : ""))",
            ),
        )

        @test Manifolds.get_parameter_type(M) === parameter

        p_random = rand(Random.Xoshiro(41), M)
        @test p_random isa VeronesePoint
        @test is_point(M, p_random)

        p_list = [[p.radial[1]], copy(p.spherical)]
        @test Manifolds.check_size(M, p_list) isa DomainError
        @test Manifolds.check_size(M, (p.radial, p.spherical)) isa DomainError

        @test Manifolds.check_size(M, invalid_points[1], X) isa DomainError
        @test_throws DomainError is_vector(M, invalid_points[1], X; error = :error)

        @testset "Equivalent representatives" begin
            p_embedded = embed(M, p)

            p_equivalent = VeronesePoint([(-1)^d * p.radial[1]], -x)
            @test embed(M, p_equivalent) ≈ p_embedded

            p_flipped = VeronesePoint([p.radial[1]], -x)
            @test embed(M, p_flipped) ≈ (-1)^d .* p_embedded
        end

        @testset "Induced metric" begin
            X_embedded = embed(M, p, X)
            Y_embedded = embed(M, p, Y)

            @test inner(M, p, X, Y) ≈ dot(X_embedded, Y_embedded)
        end

        @testset "Ambient tangent projection" begin
            A = randn(Random.Xoshiro(100 + d), n^d)
            P = project(M, p, A)

            @test P isa VeroneseTangentVector
            @test is_vector(M, p, P)
            tangent_spanning_set = [VeroneseTangentVector([1.0], zeros(n))]
            for i in 1:n
                eᵢ = zeros(n)
                eᵢ[i] = 1
                push!(tangent_spanning_set, VeroneseTangentVector([0.0], eᵢ - dot(x, eᵢ) .* x))
            end
            for W in tangent_spanning_set
                @test inner(M, p, P, W) ≈ dot(A, embed(M, p, W))
            end

            P_inplace = VeroneseTangentVector(fill(NaN, 1), fill(NaN, n))
            @test project!(M, P_inplace, p, A) === P_inplace
            @test P_inplace.radial ≈ P.radial
            @test P_inplace.spherical ≈ P.spherical

            X_projected = project(M, p, embed(M, p, X))
            @test X_projected.radial ≈ X.radial
            @test X_projected.spherical ≈ X.spherical

            d == 1 && @test embed(M, p, P) ≈ A
        end

        @testset "Zero tangent vector" begin
            p_copy = copy(M, p)
            X_copy = copy(M, p, X)
            @test p_copy == p
            @test p_copy.radial !== p.radial
            @test p_copy.spherical !== p.spherical
            @test X_copy isa VeroneseTangentVector
            @test X_copy == X
            @test X_copy.radial !== X.radial
            @test X_copy.spherical !== X.spherical

            X_zero = zero_vector(M, p)
            @test X_zero isa VeroneseTangentVector
            @test is_vector(M, p, X_zero)
            @test all(iszero, X_zero.radial)
            @test all(iszero, X_zero.spherical)

            X_inplace = deepcopy(X)
            @test zero_vector!(M, X_inplace, p) === X_inplace
            @test all(iszero, X_inplace.radial)
            @test all(iszero, X_inplace.spherical)
        end

        @testset "Representatives and distance" begin
            q_original = deepcopy(q)
            q_closest = copy(M, q)
            @test Manifolds.closest_representative!(M, q_closest, p) === q_closest
            @test embed(M, q_closest) ≈ embed(M, q)
            @test connected_by_geodesic(M, p, q)

            X_log = log(M, p, q)
            @test X_log isa VeroneseTangentVector
            q_recovered = exp(M, p, X_log)
            @test q_recovered isa VeronesePoint
            @test q_recovered.radial ≈ q_closest.radial
            @test q_recovered.spherical ≈ q_closest.spherical
            q_inplace = copy(M, p)
            @test exp!(M, q_inplace, p, X_log) === q_inplace
            @test isapprox(M, q_inplace, q_recovered)
            q_aliased = copy(M, p)
            @test exp!(M, q_aliased, q_aliased, X_log) === q_aliased
            @test isapprox(M, q_aliased, q_recovered)
            p_equivalent = VeronesePoint([(-1)^d * p.radial[1]], -p.spherical)
            X_log_equivalent =
                VeroneseTangentVector([(-1)^d * X_log.radial[1]], -X_log.spherical)
            @test embed(M, exp(M, p_equivalent, X_log_equivalent)) ≈
                embed(M, q_recovered)
            @test norm(M, p, X_log) ≈ distance(M, p, q)
            @test distance(M, p, q) ≈ distance(M, q, p)
            @test q == q_original

            q_at_zero = Manifolds.exp_fused(M, p, X, 0.0)
            @test q_at_zero.radial ≈ p.radial
            @test q_at_zero.spherical ≈ p.spherical
            t = 0.25
            q_fused = Manifolds.exp_fused(M, p, X, t)
            tX = t .* X
            @test isapprox(M, q_fused, exp(M, p, tX))
            q_fused_inplace = copy(M, p)
            @test Manifolds.exp_fused!(M, q_fused_inplace, p, X, t) === q_fused_inplace
            @test isapprox(M, q_fused_inplace, q_fused)
            @test isapprox(M, Manifolds.retract_fused(M, p, X, t), q_fused)
            @test isapprox(M, p, inverse_retract(M, p, q), X_log)

            @test connected_by_geodesic(M, p, p_equivalent)
            @test distance(M, p, p_equivalent) ≈ 0
            @test norm(M, p, log(M, p, p_equivalent)) ≈ 0 atol = 1.0e-12
            @test isapprox(M, p, p_equivalent)

            p_near = VeronesePoint([p.radial[1] + 1.0e-10], copy(p.spherical))
            @test isapprox(M, p, p_near)
            @test !isapprox(M, p, p_near; atol = 1.0e-12)

            invalid_p_copy = deepcopy(invalid_points[1])
            @test isapprox(M, invalid_points[1], invalid_p_copy)

            if iseven(d)
                q_disconnected = VeronesePoint([-p.radial[1]], q.spherical)
                @test !connected_by_geodesic(M, p, q_disconnected)
                @test isinf(distance(M, p, q_disconnected))
                @test_throws DomainError log(M, p, q_disconnected)
            end
            radial_to_zero = VeroneseTangentVector([-p.radial[1]], zeros(n))
            @test_throws DomainError exp(M, p, radial_to_zero)

            t_derivative = 1.0e-7
            q_t = Manifolds.exp_fused(M, p, X, t_derivative)
            embedded_derivative = (embed(M, q_t) - embed(M, p)) ./ t_derivative
            @test embedded_derivative ≈ embed(M, p, X) rtol = 1.0e-6 atol = 1.0e-7
        end

        @testset "Orthonormal coordinates" begin
            basis = DefaultOrthonormalBasis()
            cX = get_coordinates(M, p, X, basis)
            cY = get_coordinates(M, p, Y, basis)

            @test length(cX) == manifold_dimension(M)
            @test dot(cX, cY) ≈ inner(M, p, X, Y)
            @test norm(cX)^2 ≈ inner(M, p, X, X)

            X_roundtrip = get_vector(M, p, cX, basis)
            @test X_roundtrip isa VeroneseTangentVector
            @test X_roundtrip.radial ≈ X.radial
            @test X_roundtrip.spherical ≈ X.spherical

            cX_inplace = fill(NaN, n)
            @test get_coordinates!(M, cX_inplace, p, X, basis) === cX_inplace
            @test cX_inplace ≈ cX

            X_inplace = VeroneseTangentVector(fill(NaN, 1), fill(NaN, n))
            @test get_vector!(M, X_inplace, p, cX, basis) === X_inplace
            @test X_inplace.radial ≈ X.radial
            @test X_inplace.spherical ≈ X.spherical
        end

        @testset "Projection vector transport" begin
            transported = vector_transport_to(M, p, X, q, ProjectionTransport())
            transported_reference = project(M, q, embed(M, p, X))

            @test transported isa VeroneseTangentVector
            @test is_vector(M, q, transported)
            @test transported.radial ≈ transported_reference.radial
            @test transported.spherical ≈ transported_reference.spherical
            @test default_vector_transport_method(M) == ProjectionTransport()

            transported_inplace = VeroneseTangentVector(fill(NaN, 1), fill(NaN, n))
            @test vector_transport_to!(
                M, transported_inplace, p, X, q, ProjectionTransport()
            ) === transported_inplace
            @test transported_inplace.radial ≈ transported.radial
            @test transported_inplace.spherical ≈ transported.spherical

            q_equivalent = VeronesePoint([(-1)^d * q.radial[1]], -q.spherical)
            transported_equivalent =
                vector_transport_to(M, p, X, q_equivalent, ProjectionTransport())
            @test embed(M, q_equivalent, transported_equivalent) ≈
                embed(M, q, transported)
        end

        @testset "Embedding differential" begin
            p_embedded = embed(M, p)
            X_embedded = embed(M, p, X)

            t = 1.0e-7
            x_t = exp(Sphere(n - 1), x, t .* u)
            p_t = VeronesePoint([p.radial[1] + t * X.radial[1]], x_t)
            embedding_difference = (embed(M, p_t) - p_embedded) ./ t

            @test embedding_difference ≈ X_embedded rtol = 1.0e-6 atol = 1.0e-7
        end
    end

    @testset "VeronesePoint" begin
        p = VeronesePoint([0.5], [0.0, 1.0, 0.0])
        @test p isa VeronesePoint{Float64}
        @test Manifolds.number_eltype(p) === Float64
        @test p[1] === p.radial
        @test p[2] === p.spherical
        @test_throws BoundsError p[3]

        p_copy = copy(p)
        @test p_copy == p
        @test p_copy.radial !== p.radial
        p_copy[1] = [2.0]
        p_copy[2] = [1.0, 0.0, 0.0]
        @test p_copy == VeronesePoint([2.0], [1.0, 0.0, 0.0])
        @test copyto!(p_copy, p) === p_copy
        @test p_copy == p
        @test_throws BoundsError (p_copy[0] = [1.0])

        p_mixed = VeronesePoint(Float32[1], [0.0, 1.0])
        @test p_mixed isa VeronesePoint{Float64}
        @test Manifolds.allocate(p_mixed, Float32) isa VeronesePoint{Float32}

        p_alloc = Manifolds.allocate(p)
        @test p_alloc isa VeronesePoint{Float64}
        @test size(p_alloc.radial) == (1,)
        @test size(p_alloc.spherical) == (3,)
        @test p_alloc.radial !== p.radial
        @test p_alloc.spherical !== p.spherical
    end

    @testset "VeroneseTangentVector" begin
        X = VeroneseTangentVector([0.2], [0.0, 1.0, 0.0])
        @test X isa VeroneseTangentVector{Float64}
        @test Manifolds.number_eltype(X) === Float64
        @test X[1] === X.radial
        @test X[2] === X.spherical
        @test_throws BoundsError X[3]
        @test_throws BoundsError X[0]

        X[1] = [0.5]
        X[2] = [1.0, 0.0, -1.0]
        @test X.radial == [0.5]
        @test X.spherical == [1.0, 0.0, -1.0]
        X[1][1] = 0.7
        @test X.radial == [0.7]
        @test_throws BoundsError (X[3] = [1.0])

        X_mixed = VeroneseTangentVector(Float32[1], [0.0, 1.0])
        @test X_mixed isa VeroneseTangentVector{Float64}
        @test eltype(X_mixed.radial) === Float64

        X32 = VeroneseTangentVector(Float32[1], Float32[0, 1])
        @test 2 .* X32 isa VeroneseTangentVector{Float32}
        @test 2.0 .* X32 isa VeroneseTangentVector{Float64}

        X_alloc = Manifolds.allocate(X)
        @test X_alloc isa VeroneseTangentVector{Float64}
        @test size(X_alloc.radial) == (1,)
        @test size(X_alloc.spherical) == (3,)
        @test X_alloc.radial !== X.radial
        @test X_alloc.spherical !== X.spherical

        X_copy = copy(X)
        @test X_copy isa VeroneseTangentVector{Float64}
        @test X_copy == X
        @test X_copy.radial !== X.radial
        @test X_copy.spherical !== X.spherical
        X_copy[1][1] = 0.0
        @test X.radial == [0.7]
    end

    @testset "allocate_result for inverse_retract" begin
        M = Veronese(3, 2)
        p = VeronesePoint(Float32[0.5], Float32[0, 1, 0])
        q = VeronesePoint([0.7], [1.0, 0.0, 0.0])
        Y = ManifoldsBase.allocate_result(M, inverse_retract, p, q)
        @test Y isa VeroneseTangentVector{Float64}
        @test size(Y.radial) == (1,)
        @test size(Y.spherical) == (3,)
        Y32 = ManifoldsBase.allocate_result(M, inverse_retract, p, p)
        @test Y32 isa VeroneseTangentVector{Float32}
    end

    @testset "Geodesic obstruction at the cone apex" begin
        M = Veronese(2, 5)
        p = VeronesePoint([1.0], [1.0, 0.0])
        q = VeronesePoint([1.0], [0.0, 1.0])

        @test !connected_by_geodesic(M, p, q)
        @test distance(M, p, q) ≈ 2.0
        @test_throws DomainError log(M, p, q)
    end

    @testset "Exact geodesic cutoff" begin
        for T in (Float32, Float64)
            cutoff_cases = (
                (Veronese(2, 1), VeronesePoint(T[1], T[1, 0]), VeronesePoint(T[1], T[-1, 0])),
                (Veronese(2, 4), VeronesePoint(T[1], T[1, 0]), VeronesePoint(T[1], T[0, 1])),
            )
            for (M, p, q) in cutoff_cases
                @test !connected_by_geodesic(M, p, q)
                @test distance(M, p, q) ≈ T(2)
                @test_throws DomainError log(M, p, q)
            end

            M = Veronese(1, 1)
            p = VeronesePoint(T[1], T[1])
            q = VeronesePoint(T[-1], T[1])
            @test !connected_by_geodesic(M, p, q)
            @test isinf(distance(M, p, q))
            @test_throws DomainError log(M, p, q)
        end
    end

    @testset "Float32 rand" begin
        for parameter in (:type, :field)
            M = Veronese(3, 2; parameter)
            p = VeronesePoint(Float32[-0.5], normalize(Float32[1, 2, 3]))
            X = rand(Random.Xoshiro(12), M; vector_at = p)

            A = randn(Random.Xoshiro(13), Float32, 9)
            P = project(M, p, A)
            c = get_coordinates(M, p, X, DefaultOrthonormalBasis())
            X_roundtrip = get_vector(M, p, c, DefaultOrthonormalBasis())
            q = VeronesePoint(Float32[-0.8], normalize(Float32[-1, 1, 2]))
            Y = vector_transport_to(M, p, X, q)
            X_zero = zero_vector(M, p)
            X_log = log(M, p, q)
            q_exp = exp(M, p, X_log)
            q_fused = Manifolds.exp_fused(M, p, X, 0.25)

            for V in (X, P, X_roundtrip, Y, X_zero, X_log)
                @test V isa VeroneseTangentVector
                @test eltype(V.radial) === Float32
                @test eltype(V.spherical) === Float32
            end
            @test eltype(c) === Float32
            @test q_exp isa VeronesePoint{Float32}
            @test q_fused isa VeronesePoint{Float32}
            @test rand(Random.Xoshiro(14), M) isa VeronesePoint
            @test q_exp.radial ≈ q.radial
            @test q_exp.spherical ≈ q.spherical
        end
    end
end
