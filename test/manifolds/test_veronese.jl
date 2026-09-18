using LinearAlgebra, Manifolds, Random, Test

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

        p = ([-0.7], x)
        q = ([iseven(d) ? -0.9 : 0.9], z)
        X = ([0.2], u)
        Y = ([-0.3], v)
        Z = ([0.1], w)

        invalid_points = [
            ([1.0, 2.0], x), ([0.0], x), ([Inf], x), ([1.0], x[1:3]), ([1.0], 2 .* x),
        ]
        invalid_vectors = [
            ([0.2, 0.1], u), ([0.2], x),
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
        @test p_random isa Tuple
        @test is_point(M, p_random)

        p_list = [[p[1][1]], copy(p[2])]
        @test Manifolds.check_size(M, p_list) isa DomainError

        @test Manifolds.check_size(M, invalid_points[1], X) isa DomainError
        @test_throws DomainError is_vector(M, invalid_points[1], X; error = :error)

        @testset "Equivalent representatives" begin
            p_embedded = embed(M, p)

            p_equivalent = ([(-1)^d * p[1][1]], -x)
            @test embed(M, p_equivalent) ≈ p_embedded

            p_flipped = ([p[1][1]], -x)
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

            @test P isa Tuple
            @test is_vector(M, p, P)
            tangent_spanning_set = [([1.0], zeros(n))]
            for i in 1:n
                eᵢ = zeros(n)
                eᵢ[i] = 1
                push!(tangent_spanning_set, ([0.0], eᵢ - dot(x, eᵢ) .* x))
            end
            for W in tangent_spanning_set
                @test inner(M, p, P, W) ≈ dot(A, embed(M, p, W))
            end

            P_inplace = (fill(NaN, 1), fill(NaN, n))
            @test project!(M, P_inplace, p, A) === P_inplace
            @test P_inplace[1] ≈ P[1]
            @test P_inplace[2] ≈ P[2]

            X_projected = project(M, p, embed(M, p, X))
            @test X_projected[1] ≈ X[1]
            @test X_projected[2] ≈ X[2]

            d == 1 && @test embed(M, p, P) ≈ A
        end

        @testset "Zero tangent vector" begin
            p_copy = copy(M, p)
            X_copy = copy(M, p, X)
            @test all(p_copy[i] !== p[i] for i in eachindex(p))
            @test all(X_copy[i] !== X[i] for i in eachindex(X))

            X_zero = zero_vector(M, p)
            @test X_zero isa Tuple
            @test is_vector(M, p, X_zero)
            @test all(iszero, X_zero[1])
            @test all(iszero, X_zero[2])

            X_inplace = deepcopy(X)
            @test zero_vector!(M, X_inplace, p) === X_inplace
            @test all(iszero, X_inplace[1])
            @test all(iszero, X_inplace[2])
        end

        @testset "Representatives and distance" begin
            q_original = deepcopy(q)
            q_closest = copy(M, q)
            @test Manifolds.closest_representative!(M, q_closest, p) === q_closest
            @test embed(M, q_closest) ≈ embed(M, q)
            @test connected_by_geodesic(M, p, q)

            X_log = log(M, p, q)
            q_recovered = exp(M, p, X_log)
            @test q_recovered isa Tuple
            @test q_recovered[1] ≈ q_closest[1]
            @test q_recovered[2] ≈ q_closest[2]
            q_inplace = copy(M, p)
            @test exp!(M, q_inplace, p, X_log) === q_inplace
            @test isapprox(M, q_inplace, q_recovered)
            q_aliased = copy(M, p)
            @test exp!(M, q_aliased, q_aliased, X_log) === q_aliased
            @test isapprox(M, q_aliased, q_recovered)
            p_equivalent = ([(-1)^d * p[1][1]], -p[2])
            X_log_equivalent = ([(-1)^d * X_log[1][1]], -X_log[2])
            @test embed(M, exp(M, p_equivalent, X_log_equivalent)) ≈
                embed(M, q_recovered)
            @test norm(M, p, X_log) ≈ distance(M, p, q)
            @test distance(M, p, q) ≈ distance(M, q, p)
            @test q == q_original

            q_at_zero = Manifolds.exp_fused(M, p, X, 0.0)
            @test q_at_zero[1] ≈ p[1]
            @test q_at_zero[2] ≈ p[2]
            t = 0.25
            q_fused = Manifolds.exp_fused(M, p, X, t)
            tX = (t .* X[1], t .* X[2])
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

            p_near = ([p[1][1] + 1.0e-10], copy(p[2]))
            @test isapprox(M, p, p_near)
            @test !isapprox(M, p, p_near; atol = 1.0e-12)

            invalid_p_copy = deepcopy(invalid_points[1])
            @test isapprox(M, invalid_points[1], invalid_p_copy)

            if iseven(d)
                q_disconnected = ([-p[1][1]], q[2])
                @test !connected_by_geodesic(M, p, q_disconnected)
                @test isinf(distance(M, p, q_disconnected))
                @test_throws DomainError log(M, p, q_disconnected)
            end
            radial_to_zero = ([-p[1][1]], zeros(n))
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
            @test X_roundtrip isa Tuple
            @test X_roundtrip[1] ≈ X[1]
            @test X_roundtrip[2] ≈ X[2]

            cX_inplace = fill(NaN, n)
            @test get_coordinates!(M, cX_inplace, p, X, basis) === cX_inplace
            @test cX_inplace ≈ cX

            X_inplace = (fill(NaN, 1), fill(NaN, n))
            @test get_vector!(M, X_inplace, p, cX, basis) === X_inplace
            @test X_inplace[1] ≈ X[1]
            @test X_inplace[2] ≈ X[2]
        end

        @testset "Projection vector transport" begin
            transported = vector_transport_to(M, p, X, q, ProjectionTransport())
            transported_reference = project(M, q, embed(M, p, X))

            @test transported isa Tuple
            @test is_vector(M, q, transported)
            @test transported[1] ≈ transported_reference[1]
            @test transported[2] ≈ transported_reference[2]
            @test default_vector_transport_method(M) == ProjectionTransport()

            transported_inplace = (fill(NaN, 1), fill(NaN, n))
            @test vector_transport_to!(
                M,
                transported_inplace,
                p,
                X,
                q,
                ProjectionTransport(),
            ) === transported_inplace
            @test transported_inplace[1] ≈ transported[1]
            @test transported_inplace[2] ≈ transported[2]

            q_equivalent = ([(-1)^d * q[1][1]], -q[2])
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
            p_t = ([p[1][1] + t * X[1][1]], x_t)
            embedding_difference = (embed(M, p_t) - p_embedded) ./ t

            @test embedding_difference ≈ X_embedded rtol = 1.0e-6 atol = 1.0e-7
        end
    end

    @testset "Geodesic obstruction at the cone apex" begin
        M = Veronese(2, 5)
        p = ([1.0], [1.0, 0.0])
        q = ([1.0], [0.0, 1.0])

        @test !connected_by_geodesic(M, p, q)
        @test distance(M, p, q) ≈ 2.0
        @test_throws DomainError log(M, p, q)
    end

    @testset "Float32 rand" begin
        for parameter in (:type, :field)
            M = Veronese(3, 2; parameter)
            p = (Float32[-0.5], normalize(Float32[1, 2, 3]))
            X = rand(Random.Xoshiro(12), M; vector_at = p)

            A = randn(Random.Xoshiro(13), Float32, 9)
            P = project(M, p, A)
            c = get_coordinates(M, p, X, DefaultOrthonormalBasis())
            X_roundtrip = get_vector(M, p, c, DefaultOrthonormalBasis())
            q = (Float32[-0.8], normalize(Float32[-1, 1, 2]))
            Y = vector_transport_to(M, p, X, q)
            X_zero = zero_vector(M, p)
            X_log = log(M, p, q)
            q_exp = exp(M, p, X_log)
            q_fused = Manifolds.exp_fused(M, p, X, 0.25)

            @test X isa Tuple
            @test P isa Tuple
            @test X_roundtrip isa Tuple
            @test X_zero isa Tuple
            @test all(eltype(Xᵢ) === Float32 for Xᵢ in X)
            @test all(eltype(Pᵢ) === Float32 for Pᵢ in P)
            @test eltype(c) === Float32
            @test all(eltype(Xᵢ) === Float32 for Xᵢ in X_roundtrip)
            @test all(eltype(Yᵢ) === Float32 for Yᵢ in Y)
            @test all(eltype(Xᵢ) === Float32 for Xᵢ in X_zero)
            @test all(eltype(Xᵢ) === Float32 for Xᵢ in X_log)
            @test all(eltype(qᵢ) === Float32 for qᵢ in q_exp)
            @test all(eltype(qᵢ) === Float32 for qᵢ in q_fused)
            @test q_exp[1] ≈ q[1]
            @test q_exp[2] ≈ q[2]
        end
    end
end
