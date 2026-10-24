using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Symplectic Grassmann" begin
    @test_throws DomainError SymplecticGrassmann(2, 4)
    @test_throws ArgumentError SymplecticGrassmann(5, 2)
    @test_throws ArgumentError SymplecticGrassmann(6, 3)

    M = SymplecticGrassmann(4, 2)
    # the symplectic subspace span(e₁, e₃) rotated by the angle a towards span(e₂, e₄)
    a = 0.5
    p = [1.0 0.0; 0.0 0.0; 0.0 1.0; 0.0 0.0]
    q = [cos(a) 0.0; sin(a) 0.0; 0.0 cos(a); 0.0 sin(a)]
    # a representative of span(e₁, e₃) with r'r = diag(4, 1/4)
    r = [2.0 0.0; 0.0 0.0; 0.0 0.5; 0.0 0.0]
    # horizontal tangent vectors at p, i.e. with p⁺X = 0, vanish in the rows one and three
    E = [0.0 0.0; 1.0 0.0; 0.0 0.0; 0.0 1.0]
    X = a .* E
    Y = [0.0 0.0; 1.0 2.0; 0.0 0.0; 3.0 1.0]
    # the velocity of the rotation at q is horizontal at q, since q is orthosymplectic
    Xq = a .* [-sin(a) 0.0; cos(a) 0.0; 0.0 -sin(a); 0.0 cos(a)]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, embed, exp, geodesic, get_embedding, inner,
                inverse_retract, is_flat, is_point, is_vector, manifold_dimension, norm, repr,
                representation_size, retract, vector_transport_to, zero_vector,
            ],
            :InvalidPoints => [ones(4, 2)],
            :InvalidVectors => [ones(4, 2)],
            :InverseRetractionMethods => [CayleyInverseRetraction(), missing],
            :Points => [p, q, r],
            :RetractionMethods => [CayleyRetraction(), ExponentialRetraction()],
            :SecondVector => Y,
            :Vectors => [X, Xq, E],
            :VectorTransportMethods => [ProjectionTransport()],
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-9),
            default_inverse_retraction_method => CayleyInverseRetraction(),
            default_retraction_method => CayleyRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            (embed, :Point) => p,
            (embed, :Vector) => X,
            exp => q,
            get_embedding => SymplecticStiefel(4, 2),
            # for horizontal vectors at p with p'p = I the metric is the Euclidean one
            inner => dot(X, Y),
            # the Cayley transform of the rotation by a is (1 - a²/4, a)/(1 + a²/4), so the
            # inverse Cayley retraction of q is the tangent vector 2tan(a/2)E
            (inverse_retract, CayleyInverseRetraction()) => 2 * tan(a / 2) .* E,
            is_flat => false,
            manifold_dimension => 4,
            norm => norm(X),
            repr => "SymplecticGrassmann(4, 2)",
            representation_size => (4, 2),
            (retract, CayleyRetraction()) =>
                [1 - a^2 / 4 0.0; a 0.0; 0.0 1 - a^2 / 4; 0.0 a] ./ (1 + a^2 / 4),
            (retract, ExponentialRetraction()) => q,
            # q is orthosymplectic, so (I - qq⁺)X = (I - qqᵀ)X = cos(a) Xq
            (vector_transport_to, ProjectionTransport()) => cos(a) .* Xq,
            # the checks of the embedding are wrapped
            :IsPointErrors => [ManifoldDomainError],
            :IsVectorErrors => [ManifoldDomainError],
        ),
    )

    @testset "Basics" begin
        @test manifold_dimension(SymplecticGrassmann(6, 4)) == 8
        @test is_flat(SymplecticGrassmann(4, 4))
        # at r with r'r = diag(4, 1/4) the metric weighs the columns by (r'r)⁻¹
        @test inner(M, r, E, E) ≈ 1 / 4 + 4
        # the vertical part pΩ of a tangent vector does not contribute to the norm
        Ω = [1.0 0.0; 0.0 -1.0] # a Hamiltonian 2×2 matrix
        @test is_vector(M, p, X + p * Ω)
        @test norm(M, p, X + p * Ω) ≈ norm(M, p, X)
        @test ManifoldsBase.retract_fused(M, p, E, a, CayleyRetraction()) ≈
            [1 - a^2 / 4 0.0; a 0.0; 0.0 1 - a^2 / 4; 0.0 a] ./ (1 + a^2 / 4)
    end

    @testset "Projection onto horizontal tangent vectors" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
        # r spans span(e₁, e₃), whose symplectic projector removes the rows one and three
        Z = project(M, r, A)
        @test Z == [0.0 0.0; 3.0 4.0; 0.0 0.0; 7.0 8.0]
        @test is_vector(M, r, Z)
        @test project(M, r, Z) == Z
        # the removed part is in the kernel of the metric
        @test inner(M, r, E, A - Z) == 0
        Z2 = similar(Z)
        project!(M, Z2, r, A)
        @test Z2 == Z
        # the projection also works when the result is aliased with the input
        project!(M, A, r, A)
        @test A == Z
        # at q it removes the components along q
        @test project(M, q, X) ≈ cos(a) .* Xq
    end

    @testset "Random points" begin
        for N in (M, SymplecticGrassmann(6, 4; parameter = :field))
            @test is_point(N, rand(MersenneTwister(3), N))
            @test_throws ArgumentError rand(MersenneTwister(4), N; vector_at = rand(N))
        end
    end

    @testset "Embedding of Stiefel points and tangent vectors" begin
        pS, XS = StiefelPoint(p), StiefelTangentVector(X)
        @test embed(M, pS) == p
        @test embed(M, pS, XS) == X
        pE = similar(p)
        embed!(M, pE, pS)
        @test pE == p
        XE = similar(X)
        embed!(M, XE, pS, XS)
        @test XE == X
    end

    @testset "Riemannian gradient" begin
        A = [1.0 2.0; 0.5 -1.0; 3.0 0.0; -2.0 1.0]
        Z = riemannian_gradient(M, r, A)
        Z2 = similar(Z)
        riemannian_gradient!(M, Z2, r, A)
        @test Z2 ≈ Z
        @test is_vector(M, r, Z)
        # Z represents the differential of the Euclidean gradient A along horizontal directions
        for V in (E, [0.0 0.0; 1.0 -1.0; 0.0 0.0; 2.0 0.5])
            @test inner(M, r, Z, V) ≈ dot(A, V)
        end
    end

    @testset "Projector representation" begin
        φ(p) = p * symplectic_inverse(p)
        # the differential of φ, cf. Proposition 4.3 in BendokatZimmermann:2021
        dφ(p, X) = X * symplectic_inverse(p) + p * symplectic_inverse(X)
        pP = ProjectorPoint(φ(p))
        XP = ProjectorTangentVector(dφ(p, X))
        # the orthogonal projector onto span(e₁, e₃) and the symmetric velocity of the rotation
        @test pP.value == Diagonal([1.0, 0.0, 1.0, 0.0])
        @test XP.value == a .* [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
        Pf1 = zeros(4, 4)
        Pf1[1, 2] = 1.0 # not a projector
        Pf2 = zeros(4, 4)
        Pf2[1, 1] = 1.0 # not equal to its symplectic inverse
        Pf3 = zeros(4, 4) # of a too low rank
        Xf1 = Matrix{Float64}(I, 4, 4)
        Xf1[1, 2] = 1.0 # not Hamiltonian
        Xf2 = XP.value .+ Diagonal([1.0, 1.0, -1.0, -1.0]) # Hamiltonian, but not Xp + pX
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Functions => [embed, get_embedding, is_point, is_vector],
                :InvalidPoints => ProjectorPoint.([Pf1, Pf2, Pf3]),
                :InvalidVectors => ProjectorTangentVector.([Xf1, Xf2]),
                :Points => [pP],
                :Vectors => [XP],
            ),
            Dict(
                (embed, :Point) => pP.value,
                (embed, :Vector) => XP.value,
                get_embedding => SymplecticStiefel(4, 2),
                (get_embedding, ProjectorPoint{Matrix{Float64}}) => Euclidean(4, 4),
                :IsPointErrors => [DomainError, DomainError, DomainError],
                :IsVectorErrors => [DomainError, DomainError],
            ),
        )
        @test get_embedding(SymplecticGrassmann(4, 2; parameter = :field), typeof(pP)) ==
            Euclidean(4, 4; parameter = :field)
    end

    Manifolds.Test.test_manifold(
        SymplecticGrassmann(4, 2; parameter = :field),
        Dict(
            :Functions => [get_embedding, is_point, is_vector, manifold_dimension, repr],
            :Points => [p], :Vectors => [X],
        ),
        # Expectations
        Dict(
            get_embedding => SymplecticStiefel(4, 2; parameter = :field),
            manifold_dimension => 4,
            repr => "SymplecticGrassmann(4, 2; parameter=:field)",
        ),
    )
end
