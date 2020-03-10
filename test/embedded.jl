include("utils.jl")

struct PlaneManifold <: AbstractEmbeddedManifold{TransparentIsometricEmbedding} end

Manifolds.decorated_manifold(::PlaneManifold) = Euclidean(3)
Manifolds.base_manifold(::PlaneManifold) = Euclidean(2)

function Manifolds.check_manifold_point(M, p; kwargs...)
    length(p) != 3 && return "Wrong size"
    isapprox(p[3], 0.0; kwargs...) != 0 && return "Third component not zero"
    return nothing
end
Manifolds.embed!(::PlaneManifold, q, p) = copyto!(q, p)
Manifolds.embed!(::PlaneManifold, Y, p, X) = copyto!(Y, X)
Manifolds.project_point!(::PlaneManifold, q, p) = (q .= [p[1] p[2] 0.0])
Manifolds.project_tangent!(::PlaneManifold, Y, p, X) = (Y .= [X[1] X[2] 0.0])

struct NotImplementedEmbeddedManifold <:
       AbstractEmbeddedManifold{TransparentIsometricEmbedding} end
Manifolds.decorated_manifold(::NotImplementedEmbeddedManifold) = Euclidean(2)
Manifolds.base_manifold(::NotImplementedEmbeddedManifold) = Euclidean(2)

struct NotImplementedEmbeddedManifold2 <:
       AbstractEmbeddedManifold{AbstractIsometricEmbeddingType} end

struct NotImplementedEmbeddedManifold3 <: AbstractEmbeddedManifold{AbstractEmbeddingType} end

@testset "Embedded Manifolds" begin
    @testset "EmbeddedManifold basic tests" begin
        M = EmbeddedManifold(Euclidean(2), Euclidean(3))
        @test repr(M) ==
              "EmbeddedManifold(Euclidean(2; field = ℝ), Euclidean(3; field = ℝ), TransparentIsometricEmbedding())"
        @test decorated_manifold(M) == Euclidean(3)
        @test Manifolds.default_embedding_dispatch(M) === Val{false}()
        @test ManifoldsBase.default_decorator_dispatch(M) === Manifolds.default_embedding_dispatch(M)
    end
    @testset "PlaneManifold" begin
        M = PlaneManifold()
        @test repr(M) == "PlaneManifold()"
        @test ManifoldsBase.default_decorator_dispatch(M) === Val{false}()
        p = [1.0 1.0 0.0]
        q = [1.0 0.0 0.0]
        X = q - p
        @test embed(M, p) == p
        pE = similar(p)
        embed!(M, pE, p)
        @test pE == p
        P = [1.0 1.0 2.0]
        Q = similar(P)
        @test project_point!(M, Q, P) == project_point!(M, Q, P)
        @test project_point!(M, Q, P) == [1.0 1.0 0.0]

        @test log(M, p, q) == q - p
        Y = similar(p)
        log!(M, Y, p, q)
        @test Y == q - p
        @test exp(M, p, X) == q
        r = similar(p)
        exp!(M, r, p, X)
        @test r == q

    end

    @testset "Test nonimplemented fallbacks" begin
        @testset "Default Isometric Embedding Fallback Error Tests" begin
            M = NotImplementedEmbeddedManifold()
            A = zeros(2)
            @test_throws ErrorException check_manifold_point(M, [1, 2])
            @test_throws ErrorException check_tangent_vector(M, [1, 2], [3, 4])
            @test norm(M, [1, 2], [2, 3]) ≈ sqrt(13)
            @test inner(M, [1, 2], [2, 3], [2, 3]) ≈ 13
            @test_throws ErrorException manifold_dimension(M)
            # without any implementation the projections are the identity
            @test project_point(M, [1, 2]) == [1, 2]
            @test project_tangent(M, [1, 2], [2, 3]) == [2, 3]
            project_tangent!(M, A, [1, 2], [2, 3])
            @test A == [2, 3]
            @test vector_transport_direction(M, [1, 2], [2, 3], [3, 4]) == [2, 3]
            vector_transport_direction!(M, A, [1, 2], [2, 3], [3, 4])
            @test A == [2, 3]
            @test vector_transport_to(M, [1, 2], [2, 3], [3, 4]) == [2, 3]
            vector_transport_to!(M, A, [1, 2], [2, 3], [3, 4])
            @test A == [2, 3]
        end
        @testset "General Isometric Embedding Fallback Error Tests" begin
            M2 = NotImplementedEmbeddedManifold2()
            A = zeros(2)
            @test_throws ErrorException exp(M2, [1, 2], [2, 3])
            @test_throws ErrorException exp!(M2, A, [1, 2], [2, 3])
            @test_throws ErrorException log(M2, [1, 2], [2, 3])
            @test_throws ErrorException log!(M2, A, [1, 2], [2, 3])
            @test_throws ErrorException manifold_dimension(M2)
            @test_throws ErrorException project_point(M2, [1, 2])
            @test_throws ErrorException project_point!(M2, A, [1, 2])
            @test_throws ErrorException project_tangent(M2, [1, 2], [2, 3])
            @test_throws ErrorException project_tangent!(M2, A, [1, 2], [2, 3])
            @test_throws ErrorException vector_transport_along(M2, [1, 2], [2, 3], [])
            @test_throws ErrorException vector_transport_along!(M2, A, [1, 2], [2, 3], [])
            @test_throws ErrorException vector_transport_direction(
                M2,
                [1, 2],
                [2, 3],
                [3, 4],
            )
            @test_throws ErrorException vector_transport_direction!(
                M2,
                A,
                [1, 2],
                [2, 3],
                [3, 4],
            )
            @test_throws ErrorException vector_transport_to(M2, [1, 2], [2, 3], [3, 4])
            @test_throws ErrorException vector_transport_to!(M2, A, [1, 2], [2, 3], [3, 4])
        end
        @testset "Nonisometric Embedding Fallback Error Rests" begin
            M3 = NotImplementedEmbeddedManifold3()
            @test_throws ErrorException inner(M3, [1, 2], [2, 3], [2, 3])
            @test_throws ErrorException manifold_dimension(M3)
            @test_throws ErrorException norm(M3, [1, 2], [2, 3])
            @test_throws ErrorException embed(M3, [1, 2], [2, 3])
            @test_throws ErrorException embed(M3, [1, 2])
        end
    end
end
