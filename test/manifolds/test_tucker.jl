using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test
using RecursiveArrayTools

@testset "Tucker" begin
    @test_throws DomainError Tucker((2, 3), (3, 3))
    @test_throws DomainError Tucker((4, 5, 6), (1, 2, 3))

    M = Tucker((3, 3, 3), (2, 2, 2))
    # canonical basis vectors and their tensor products
    e(i) = [j == i ? 1.0 : 0.0 for j in 1:3]
    ⊗(a, b, c) = [a[i] * b[j] * c[k] for i in 1:3, j in 1:3, k in 1:3]
    δ(i, j, k) = (D = zeros(2, 2, 2); D[i, j, k] = 1.0; D)
    # a point in HOSVD form with factors U, core C and singular values σ of its unfoldings
    tucker_point(C, U...) = TuckerPoint(Manifolds.HOSVD{Float64, 3}(U, C, ntuple(_ -> [C[1, 1, 1], C[2, 2, 2]], 3)))
    E = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    Z = zeros(3, 2)
    # p = 2 e₁⊗e₁⊗e₁ + e₂⊗e₂⊗e₂, whose unfoldings have the singular values 2 and 1
    C = 2 .* δ(1, 1, 1) .+ δ(2, 2, 2)
    p = tucker_point(C, E, E, E)
    # X rotates the first factor of the first term towards e₃, Y also changes the core
    t = 0.5
    X = TuckerTangentVector(zeros(2, 2, 2), ([0.0 0.0; 0.0 0.0; t 0.0], Z, Z))
    Y = TuckerTangentVector(δ(1, 2, 1), ([0.0 0.0; 0.0 0.0; 1.0 1.0], Z, Z))
    W = TuckerTangentVector(δ(2, 1, 1), (Z, [0.0 0.0; 0.0 0.0; 0.0 1.0], Z))
    # p + X = 2s u⊗e₁⊗e₁ + e₂⊗e₂⊗e₂ with u = (e₁ + te₃)/s has the same multilinear rank
    s = sqrt(1 + t^2)
    u = (e(1) .+ t .* e(3)) ./ s
    w = (e(3) .- t .* e(1)) ./ s # completes u to a basis of span(e₁, e₃)
    q = tucker_point(2s .* δ(1, 1, 1) .+ δ(2, 2, 2), [u e(2)], E, E)
    # invalid points: of a wrong size, with singular values that do not match the core,
    # and of a too low rank
    pf1 = TuckerPoint(C, [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])
    pf2 = TuckerPoint(Manifolds.HOSVD{Float64, 3}((E, E, E), C, ntuple(_ -> [1.0, 1.0], 3)))
    pf3 = TuckerPoint(zeros(2, 2, 2), E, E, E)
    # invalid vectors: a factor that is not orthogonal to that of p, and one of wrong size
    Xf1 = TuckerTangentVector(zeros(2, 2, 2), (E, Z, Z))
    Xf2 = TuckerTangentVector(zeros(2, 2, 2), (zeros(2, 2), Z, Z))
    B = DefaultOrthonormalBasis()
    # the coordinates of the core come first, then those of the factors scaled by σ
    cX = [zeros(8); 2t; zeros(5)]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Bases => [B],
            :Coordinates => [cX],
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, embed, get_coordinates, get_embedding,
                get_vector, get_vectors, inner, inverse_retract, is_flat, is_point, is_vector,
                manifold_dimension, norm, rand, repr, retract, vector_transport_to,
                zero_vector,
            ],
            :InvalidPoints => [pf1, pf2, pf3],
            :InvalidVectors => [Xf1, Xf2],
            :InverseRetractionMethods => [ProjectionInverseRetraction()],
            :Points => [p, q],
            :RetractionMethods => [PolarRetraction()],
            :SecondVector => Y,
            :Vectors => [X, Y, W],
            :VectorTransportMethods => [ProjectionTransport()],
            :Seed => 42,
            :TestRandScaling => true,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            default_inverse_retraction_method => ProjectionInverseRetraction(),
            default_retraction_method => PolarRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            (embed, :Point) => 2 .* ⊗(e(1), e(1), e(1)) .+ ⊗(e(2), e(2), e(2)),
            (embed, :Vector) => 2t .* ⊗(e(3), e(1), e(1)),
            (get_coordinates, B) => cX,
            get_embedding => Euclidean(3, 3, 3),
            (get_vector, cX, B) => X,
            (get_vectors, B) => :Orthonormal,
            # the factors are weighted by the squared singular values 4 and 1
            inner => 4t,
            # p + X is a point, so its difference to p is tangent
            (inverse_retract, ProjectionInverseRetraction()) => X,
            is_flat => false,
            manifold_dimension => 14,
            norm => 2t,
            repr => "Tucker((3, 3, 3), (2, 2, 2), ℝ)",
            (retract, PolarRetraction()) => q,
            # X = 2t e₃⊗e₁⊗e₁ = 2t²/s u⊗e₁⊗e₁ + 2t/s w⊗e₁⊗e₁ is also tangent at q
            (vector_transport_to, ProjectionTransport()) =>
                TuckerTangentVector(2t^2 / s .* δ(1, 1, 1), ([w zeros(3)] .* t / s^2, Z, Z)),
            :IsPointErrors => [DomainError, DomainError, DomainError],
            :IsVectorErrors => [DomainError, DomainError],
            :IsVectorBasepointError => DomainError,
        ),
    )

    @testset "Basics" begin
        @test manifold_dimension(Tucker((4, 5, 6), (2, 3, 4))) == 24 + 2 * 2 + 3 * 2 + 4 * 2
        # the projection of a tensor keeps the parts that are linear in the factors and the core
        A = ⊗(e(1), e(2), e(1)) .+ ⊗(e(3), e(1), e(1)) .+ ⊗(e(3), e(3), e(3))
        PA = TuckerTangentVector(δ(1, 2, 1), ([0.0 0.0; 0.0 0.0; 0.5 0.0], Z, Z))
        @test isapprox(M, p, project(M, p, A), PA)
        @test isapprox(M, p, project!(M, zero_vector(M, p), p, A), PA)
        # inner products with tensors in the embedding
        @test inner(M, p, X, embed(M, p, Y)) ≈ 4t
        @test inner(M, p, embed(M, p, Y), X) ≈ 4t
        # full arrays are points if they have the right size and multilinear rank
        @test is_point(M, embed(M, p))
        @test !is_point(M, zeros(3, 3, 3))
        @test !is_point(M, zeros(3, 3, 2))
        @test_throws DomainError is_point(M, ⊗(e(1), e(1), e(1)); error = :error)
        @test !is_point(Tucker((2, 2, 2), (2, 2, 2)), p)
        @test_throws DomainError is_point(Tucker((3, 3, 3), (1, 2, 2)), p; error = :error)
        # scaling the core breaks the match with the singular values
        pa = allocate(p)
        copyto!(M, pa, p)
        @test pa == p
        pa.hosvd.core .*= 2
        @test !is_point(M, pa)
        @test !is_vector(M, q, X)
        # truncating the HOSVD of the full tensor recovers p
        @test isapprox(M, p, TuckerPoint(embed(M, p), (2, 2, 2)))
        @test isapprox(p, TuckerPoint(C, E, E, E))
        # the factors of tangent vectors are orthogonal to those of the point up to a tolerance
        Xs = TuckerTangentVector(zeros(2, 2, 2), ([1.0e-6 0.0; 0.0 0.0; t 0.0], Z, Z))
        @test !is_vector(M, p, Xs)
        @test is_vector(M, p, Xs; atol = 1.0e-5)
        # the coordinates do not depend on the default basis
        @test get_coordinates(M, p, X, DefaultOrthogonalBasis()) ≈ cX
        @test get_coordinates(M, p, X, DefaultBasis()) ≈ cX
        @test get_coordinates!(M, similar(cX), p, X, get_basis(M, p, B)) ≈ cX
        @test get_coordinates!(M, similar(cX), p, X) ≈ cX
    end

    @testset "Tangent vectors" begin
        @test +X == X
        @test -X == TuckerTangentVector(-X.Ċ, map(-, X.U̇))
        @test X / 2 == 2 \ X == 0.5 * X == X * 0.5
        @test axes(X) === ()
        @test size(X) === ()
        @test length(X) === 1
        Xc = copy(X)
        @test Xc == X
        @test Xc.U̇[1] !== X.U̇[1]
        V = Y .+ X .* 2
        @test V isa TuckerTangentVector
        @test V == Y + 2 * X
        V .= 2 .* Y .+ X
        @test V == 2 * Y + X
        pc = copy(p)
        @test pc == p
        @test pc.hosvd.U[1] !== p.hosvd.U[1]
    end

    @testset "Basis" begin
        ℬ = get_basis(M, p, B)
        J = convert(Matrix, ℬ)
        # the basis vectors are the tensor products of basis vectors of the factors, and
        # the variations of the factors towards e₃ weighted by the inverse singular values
        @test J[:, 1] == vec(⊗(e(1), e(1), e(1)))
        @test J[:, 9] == vec(⊗(e(3), e(1), e(1)))
        @test J[:, 10] == vec(⊗(e(3), e(2), e(2)))
        @test J'J ≈ I
        @test J ≈ reduce(hcat, map(V -> vec(embed(M, p, V)), get_vectors(M, p, ℬ)))
    end

    @testset "show" begin
        shiftprint(x) = replace(sprint(show, "text/plain", x), "\n" => "\n ")
        @test sprint(show, "text/plain", p) == """
            $(summary(p))
            U factor 1:
             $(shiftprint(p.hosvd.U[1]))\n
            U factor 2:
             $(shiftprint(p.hosvd.U[2]))\n
            U factor 3:
             $(shiftprint(p.hosvd.U[3]))\n
            Core:
             $(shiftprint(p.hosvd.core))"""
        @test sprint(show, "text/plain", X) == """
            $(summary(X))
            U̇ factor 1:
             $(shiftprint(X.U̇[1]))\n
            U̇ factor 2:
             $(shiftprint(X.U̇[2]))\n
            U̇ factor 3:
             $(shiftprint(X.U̇[3]))\n
            Ċ factor:
             $(shiftprint(X.Ċ))"""
        ℬ = get_basis(M, p, B)
        @test sprint(show, "text/plain", ℬ) == """
            $(summary(ℬ)) ≅ $(shiftprint(convert(Matrix, ℬ)))
            """
    end

    Manifolds.Test.test_manifold(
        Tucker((3, 3, 3), (2, 2, 2); parameter = :field),
        Dict(
            :Functions => [get_embedding, is_point, is_vector, manifold_dimension, repr],
            :Points => [p], :Vectors => [X],
        ),
        # Expectations
        Dict(
            get_embedding => Euclidean(3, 3, 3; parameter = :field),
            manifold_dimension => 14,
            repr => "Tucker((3, 3, 3), (2, 2, 2), ℝ; parameter=:field)",
        ),
    )

    # the transport implements the projection of the embedded tangent vector at q
    # in the factors of the points, here at points of different multilinear ranks
    @testset "projection transport" begin
        M2 = Tucker((4, 5, 6), (2, 3, 4))
        p2 = TuckerPoint(reshape([mod(17k, 23) / 23 for k in 1:120], 4, 5, 6), (2, 3, 4))
        q2 = TuckerPoint(reshape([mod(13k, 29) / 29 for k in 1:120], 4, 5, 6), (2, 3, 4))
        X2 = project(M2, p2, reshape([mod(3k, 7) / 7 for k in 1:120], 4, 5, 6))
        @test isapprox(
            M2, q2, vector_transport_to(M2, p2, X2, q2, ProjectionTransport()),
            project(M2, q2, embed(M2, p2, X2)),
        )
    end

    @testset "projection transport on complex tensors" begin
        Mc = Tucker((4, 5, 6), (2, 3, 4), ℂ)
        @test get_embedding(Mc) == Euclidean(4, 5, 6; field = ℂ)
        A = reshape([complex(mod(17k, 23) / 23, mod(11k, 19) / 19) for k in 1:120], 4, 5, 6)
        Bc = reshape([complex(mod(13k, 29) / 29, mod(7k, 31) / 31) for k in 1:120], 4, 5, 6)
        pc = TuckerPoint(A, (2, 3, 4))
        qc = TuckerPoint(Bc, (2, 3, 4))
        U̇ = map(pc.hosvd.U) do U
            n = size(U, 1)
            Zc = reshape([complex(mod(3k, 7) / 7, mod(2k, 5) / 5) for k in 1:(n^2)], n, n)
            return Zc * U - U * (U' * Zc * U)
        end
        Xc = TuckerTangentVector((0.5 + 0.25im) * pc.hosvd.core, U̇)
        Yc = vector_transport_to(Mc, pc, Xc, qc, ProjectionTransport())
        # the columns of J are an orthonormal basis of the complex tangent space at qc
        J = convert(Matrix, get_basis(Mc, qc, DefaultOrthonormalBasis()))
        @test vec(embed(Mc, qc, Yc)) ≈ J * (J' * vec(embed(Mc, pc, Xc)))
    end

    @testset "product of Tucker manifolds" begin
        MP = ProductManifold(Tucker((18, 16, 14), (4, 3, 3)), Tucker((18, 16, 14), (4, 3, 3)))
        pP = rand(MersenneTwister(42), MP)
        XP = rand(MersenneTwister(42), MP; vector_at = pP)
        YP = copy(MP, XP)
        YP .*= 2
        @test isapprox(MP, pP, XP + XP, YP)
    end
end
