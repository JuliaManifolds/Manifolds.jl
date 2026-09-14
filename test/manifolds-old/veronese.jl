using LinearAlgebra, Manifolds, Random, Test

@testset "Veronese Manifold" begin
    @testset "Veronese" begin
        @test Veronese(4, 3) == Veronese{4, 3}()
        @test sprint(show, Veronese(4, 3)) == "Veronese(4, 3)"
        @test_throws ArgumentError Veronese(0, 2)
        @test_throws ArgumentError Veronese(3, 0)
    end

    n = 4

    x = normalize([1.0, 2.0, -1.0, 0.5])
    u = normalize([x[2], -x[1], 0.0, 0.0])

    v_raw = [0.5, -1.0, 2.0, 1.0]
    v = v_raw - dot(x, v_raw) * x

    Ms = [Veronese(n, d) for d in 1:3]
    Ds = 1:3

    ps = [[[-0.7], x] for _ in Ds]
    Xs = [[[0.2], u] for _ in Ds]
    Ys = [[[-0.3], v] for _ in Ds]

    for (M, d, p, X, Y) in zip(Ms, Ds, ps, Xs, Ys)
        @testset "Manifold $M" begin
            @testset "manifold_dimension" begin
                @test manifold_dimension(M) == n
            end

            @testset "is_point" begin
                @test is_point(M, p)

                @test_throws DomainError is_point(
                    M,
                    [[0.0], x];
                    error = :error,
                )
                @test_throws DomainError is_point(
                    M,
                    [[Inf], x];
                    error = :error,
                )
                @test_throws DomainError is_point(
                    M,
                    [[1.0, 2.0], x];
                    error = :error,
                )
                @test_throws DomainError is_point(
                    M,
                    [[1.0], x[1:3]];
                    error = :error,
                )
                @test_throws DomainError is_point(
                    M,
                    [[1.0], 2 .* x];
                    error = :error,
                )
            end

            @testset "is_vector" begin
                @test is_vector(M, p, X)
                @test is_vector(M, p, Y)

                @test_throws DomainError is_vector(
                    M,
                    p,
                    [[0.2, 0.1], u];
                    error = :error,
                )
                @test_throws DomainError is_vector(
                    M,
                    p,
                    [[0.2], x];
                    error = :error,
                )
            end

            @testset "rand" begin
                rng = Random.Xoshiro(42)

                random_p = rand(rng, M)
                random_X = rand(rng, M; vector_at = p)

                @test is_point(M, random_p)
                @test is_vector(M, p, random_X)
            end

            @testset "get_embedding" begin
                @test get_embedding(M) == Euclidean(n^d)
            end

            @testset "embed!" begin
                # point
                p_embedded = zeros(n^d)
                embed!(M, p_embedded, p)

                @test is_point(get_embedding(M), p_embedded)

                # equivalent representative
                p_equivalent = [[(-1)^d * p[1][1]], -x]
                p_equivalent_embedded = zeros(n^d)
                embed!(M, p_equivalent_embedded, p_equivalent)

                @test p_equivalent_embedded ≈ p_embedded

                # flip x without changing lambda
                p_flipped_embedded = zeros(n^d)
                embed!(M, p_flipped_embedded, [[p[1][1]], -x])

                @test p_flipped_embedded ≈ (-1)^d .* p_embedded

                # tangent vectors
                X_embedded = zeros(n^d)
                Y_embedded = zeros(n^d)

                embed!(M, X_embedded, p, X)
                embed!(M, Y_embedded, p, Y)

                @test is_vector(get_embedding(M), p_embedded, X_embedded)
                @test is_vector(get_embedding(M), p_embedded, Y_embedded)
            end

            @testset "inner" begin
                X_embedded = embed(M, p, X)
                Y_embedded = embed(M, p, Y)

                @test inner(M, p, X, Y) ≈ dot(X_embedded, Y_embedded)
            end

            @testset "embedding differential" begin
                p_embedded = embed(M, p)
                X_embedded = embed(M, p, X)

                t = 1.0e-7
                x_t = exp(Sphere(n - 1), x, t .* u)
                p_t = [[p[1][1] + t * X[1][1]], x_t]

                embedding_difference =
                    (embed(M, p_t) - p_embedded) ./ t

                @test embedding_difference ≈ X_embedded rtol = 1.0e-6 atol = 1.0e-7
            end
        end
    end

    @testset "Float32 rand" begin
        M = Veronese(3, 2)
        p = [Float32[-0.5], normalize(Float32[1, 2, 3])]

        X = rand(Random.Xoshiro(12), M; vector_at = p)

        @test all(eltype(Xᵢ) === Float32 for Xᵢ in X)
    end
end
