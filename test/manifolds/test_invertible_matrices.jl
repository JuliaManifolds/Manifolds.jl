using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The manifold of invertible matrices" begin
    @test_throws DomainError InvertibleMatrices(-1)

    @testset "Real" begin
        M = InvertibleMatrices(3, ℝ)
        p1 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        p2 = [2.0 1.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        p3 = [1.0 0.0 0.0; 1.0 3.0 0.0; 0.0 0.0 -1.0]
        X = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
        X2 = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        Y = [0.5 0.0 -1.0; 0.0 2.0 0.0; 1.0 0.0 0.0]
        # Invalids
        pf1 = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] # singular
        pf2 = [1.0im 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        Xf1 = [1.0im 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        bases = [DefaultOrthogonalBasis(), DefaultOrthonormalBasis()]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => bases,
                :Coordinates => [vec(X), vec(X)],
                :Functions => [
                    copy, copyto!, embed, get_coordinates, get_embedding, get_vector,
                    is_flat, is_point, is_vector, manifold_dimension, rand, repr,
                    representation_size, sectional_curvature, Weingarten, zero_vector,
                ],
                :InvalidPoints => AbstractArray[pf1, pf2], # AbstractArray avoids promotion to complex
                :InvalidVectors => [Xf1],
                :NormalVectors => [zero(p1)],
                :Points => [p1, p2, p3],
                :SecondVector => X2,
                :Vectors => [X, Y, X2],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                (embed, :Point) => p1,
                (embed, :Vector) => X,
                (get_coordinates, bases[1]) => vec(X),
                (get_coordinates, bases[2]) => vec(X),
                (get_vector, vec(X), bases[1]) => X,
                (get_vector, vec(X), bases[2]) => X,
                get_embedding => Euclidean(3, 3),
                is_flat => true,
                manifold_dimension => 9,
                repr => "InvertibleMatrices(3, ℝ)",
                representation_size => (3, 3),
                sectional_curvature => 0.0,
                sectional_curvature_max => 0.0,
                sectional_curvature_min => 0.0,
                Weingarten => zero(p1),
                :IsPointErrors => [DomainError, ManifoldDomainError],
                :IsVectorErrors => [ManifoldDomainError],
            ),
        )

        @testset "Further properties" begin
            @test embed(M, p1) === p1
            @test embed(M, p1, X) === X
            @test riemann_tensor(M, p1, X, X2, Y) == zero(p1)
            @test sectional_curvature(M, p1, X, X2) === 0.0
            M2 = InvertibleMatrices(3, ℝ; parameter = :field)
            @test repr(M2) == "InvertibleMatrices(3, ℝ; parameter=:field)"
            @test typeof(get_embedding(M)) ===
                Euclidean{ℝ, ManifoldsBase.TypeParameter{Tuple{3, 3}}}
            @test typeof(get_embedding(M2)) === Euclidean{ℝ, Tuple{Int64, Int64}}
        end

        @testset "tangent projection" begin
            # points are not projected, since the manifold is an open subset of its embedding
            @test project(M, p1, X) == X
            Z = zero(X)
            @test project!(M, Z, p1, X) === Z
            @test Z == X
        end
    end

    @testset "Complex" begin
        M = InvertibleMatrices(3, ℂ)
        p1 = [1.0im 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        p2 = ComplexF64[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        p3 = [2.0 1.0im 0.0; 0.0 1.0 0.0; 0.0 0.0 -1.0im]
        X = [1.0 2.0im 3.0; 4.0 5.0 6.0im; 7.0 8.0 10.0]
        X2 = 1.0im .* p1
        Y = [0.5im 0.0 -1.0; 0.0 2.0 0.0; 1.0 0.0 1.0im]
        pf1 = [0.0im 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] # singular
        pf2 = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] # singular and real
        bases = [DefaultOrthogonalBasis(), DefaultOrthonormalBasis(), DefaultOrthonormalBasis(ℂ)]
        cX = [real(vec(X)); imag(vec(X))]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => bases,
                :Coordinates => [
                    get_coordinates(M, p1, X, bases[1]), cX,
                    get_coordinates(M, p1, X, bases[3]),
                ],
                :Functions => [
                    copy, copyto!, embed, get_coordinates, get_embedding, get_vector,
                    is_flat, is_point, is_vector, manifold_dimension, rand, repr,
                    representation_size, sectional_curvature, zero_vector,
                ],
                :InvalidPoints => AbstractArray[pf1, pf2],
                :Points => [p1, p2, p3],
                :SecondVector => X2,
                :Vectors => [X, Y, X2],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                (get_coordinates, bases[2]) => cX,
                (get_vector, cX, bases[2]) => X,
                get_embedding => Euclidean(3, 3; field = ℂ),
                is_flat => true,
                manifold_dimension => 18,
                (number_eltype, rand) => ComplexF64,
                repr => "InvertibleMatrices(3, ℂ)",
                representation_size => (3, 3),
                sectional_curvature => 0.0,
                :IsPointErrors => [DomainError, DomainError],
            ),
        )

        @testset "Further properties" begin
            # real matrices are valid points as well
            @test is_point(M, real(p2); error = :error)
            Mc2 = InvertibleMatrices(3, ℂ; parameter = :field)
            @test repr(Mc2) == "InvertibleMatrices(3, ℂ; parameter=:field)"
            @test project(M, p1, X) == X
        end
    end
end
