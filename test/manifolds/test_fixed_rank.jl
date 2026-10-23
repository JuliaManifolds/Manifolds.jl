using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Fixed Rank Matrices" begin
    @test_throws DomainError FixedRankMatrices(3, 2, 3)
    @test_throws DomainError FixedRankMatrices(3, 2, -1)

    M = FixedRankMatrices(3, 2, 2)
    pE = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    p = SVDMPoint(pE)
    X = UMVTangentVector([0.0 0.0; 0.0 0.0; 1.0 1.0], [1.0 0.0; 0.0 1.0], zeros(2, 2))
    X2 = UMVTangentVector(X.U, 2 * X.M, X.Vt)
    q = retract(M, p, X, PolarRetraction())
    r = SVDMPoint([1 / sqrt(2) 1 / sqrt(2); 1 / sqrt(2) -1 / sqrt(2); 0.0 0.0])
    Y = project(M, q, embed(M, p, X))
    Z = project(M, r, [0.5 -1.0; 0.25 0.0; 1.0 2.0])
    # Invalids
    pf1 = SVDMPoint([1.0 0.0; 0.0 0.0], 2) # wrong size
    pf2 = SVDMPoint([1.0 0.0; 0.0 1.0; 0.0 0.0], [1.0, 0.0], [1.0 0.0; 0.0 1.0]) # rank 1
    pf3 = SVDMPoint(2 * p.U, p.S, p.Vt) # U not orthonormal
    pf4 = SVDMPoint(p.U, p.S, 2 * p.Vt) # Vt not orthonormal
    Xf1 = UMVTangentVector(p.U, X.M, p.Vt, 2) # U not orthogonal to p.U
    Xf2 = UMVTangentVector(X.U, X.M, p.Vt, 2) # Vt not orthogonal to p.Vt
    Xf3 = UMVTangentVector(zeros(2, 1), zeros(1, 2), zeros(2, 2)) # wrong size

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, embed, get_embedding, injectivity_radius,
                inner, inverse_retract, is_default_metric, is_flat, is_point, is_vector,
                manifold_dimension, norm, rand, repr, representation_size, retract,
                zero_vector,
            ],
            :Points => [p, q, r], :Vectors => [X, Y, Z], :SecondVector => X2,
            :InvalidPoints => [pf1, pf2, pf3, pf4],
            :InvalidVectors => [Xf1, Xf2, Xf3],
            :RetractionMethods => [PolarRetraction(), OrthographicRetraction()],
            :InverseRetractionMethods => [missing, OrthographicInverseRetraction()],
            :Seed => 42,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-13,
            default_inverse_retraction_method => OrthographicInverseRetraction(),
            default_retraction_method => OrthographicRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            embed => pE,
            (embed, :Vector) => p.U * X.M * p.Vt + X.U * p.Vt + p.U * X.Vt,
            get_embedding => Euclidean(3, 2),
            injectivity_radius => 0.0,
            inner => 6.0,
            is_default_metric => EuclideanMetric(),
            is_flat => false,
            manifold_dimension => 6,
            norm => 2.0,
            repr => "FixedRankMatrices(3, 2, 2, ℝ)",
            representation_size => (3, 2),
            (retract, OrthographicRetraction()) => SVDMPoint([2.0 0.0; 0.0 2.0; 1.0 1.0]),
            :IsPointErrors => [DomainError, DomainError, DomainError, DomainError],
            :IsVectorErrors => [DomainError, DomainError, missing],
            :IsVectorBasepointError => ManifoldDomainError,
        ),
    )

    # Tested separately: the `UMVTangentVector` representation depends on the SVD factors
    # of its base point, so the transport `to` and `direction` are only comparable
    # if both end at the identical `SVDMPoint`, here the default retraction of `X`.
    qX = retract(M, p, X)
    Manifolds.Test.test_vector_transport(
        M, p, X, qX, ProjectionTransport();
        available_functions = [vector_transport_direction],
        direction = X,
        expected_value = project(M, qX, embed(M, p, X)),
        expected_value_direction = project(M, qX, embed(M, p, X)),
        name = "vector_transport_to(M, p, X, q, ProjectionTransport())",
        atol = 1.0e-13,
    )

    @testset "show" begin
        @test sprint(show, "text/plain", p) == """
            $(sprint(show, SVDMPoint{Matrix{Float64}, Vector{Float64}, Matrix{Float64}}))
            U factor:
             3×2 $(sprint(show, Matrix{Float64})):
              1.0  0.0
              0.0  1.0
              0.0  0.0
            singular values:
             2-element $(sprint(show, Vector{Float64})):
              1.0
              1.0
            Vt factor:
             2×2 $(sprint(show, Matrix{Float64})):
              1.0  0.0
              0.0  1.0"""
        @test sprint(show, "text/plain", X) == """
            $(sprint(show, UMVTangentVector{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}))
            U factor:
             3×2 $(sprint(show, Matrix{Float64})):
              0.0  0.0
              0.0  0.0
              1.0  1.0
            M factor:
             2×2 $(sprint(show, Matrix{Float64})):
              1.0  0.0
              0.0  1.0
            Vt factor:
             2×2 $(sprint(show, Matrix{Float64})):
              0.0  0.0
              0.0  0.0"""
    end

    @testset "Basics" begin
        @test inner(M, p, X, X) == norm(M, p, X)^2
        @test metric(M) == EuclideanMetric()
        @test is_point(M, q)
        @test is_point(M, r)
        @test !is_vector(M, pf1, X)
        # a point is an m×n matrix of rank exactly k, also in SVD form
        @test is_point(M, pE)
        @test !is_point(M, [1.0 0.0; 0.0 0.0; 0.0 0.0])
        @test !is_point(M, pE[1:2, :])
        @test_throws DomainError is_point(M, pE[1:2, :]; error = :error)
        @test_throws DomainError is_point(FixedRankMatrices(3, 2, 1), p; error = :error)
        @test_throws DomainError is_point(FixedRankMatrices(3, 2, 1), pE; error = :error)
        # the embedding of a point is a valid (matrix) point
        @test is_point(M, embed(M, q))
        @test isapprox(get_embedding(M), embed(M, q), q.U * Diagonal(q.S) * q.Vt)
        # tangent projection
        B = embed(M, p, X)
        BB = similar(B)
        embed!(M, BB, p, X)
        @test isapprox(M, p, B, BB)
        @test isapprox(M, p, X, project(M, p, B))
        X3 = project(M, p, B)
        project!(M, X3, p, 2 .* B)
        @test isapprox(M, p, X3, 2 * X)
        # projection transport is the projection of the embedded vector
        @test isapprox(
            M, q, vector_transport_to(M, p, X, q, ProjectionTransport()),
            project(M, q, embed(M, p, X)),
        )
    end

    @testset "Rank 1" begin
        M2 = FixedRankMatrices(3, 2, 1)
        p2 = SVDMPoint([1.0 0.0; 0.0 1.0; 0.0 0.0], 1)
        Manifolds.Test.test_manifold(
            M2,
            Dict(
                :Functions => [injectivity_radius, is_point, manifold_dimension],
                :Points => [p2],
                :InvalidPoints => [pE],
            ),
            # Expectations
            Dict(
                injectivity_radius => 0.0,
                manifold_dimension => 4,
                :IsPointErrors => [DomainError],
            ),
        )
        @test Manifolds.check_point(M2, pE) isa DomainError
        @test is_point(M2, [1.0 0.0; 0.0 0.0; 0.0 0.0])
    end

    @testset "Tangent vectors on FixedRankMatrices(4, 3, 2)" begin
        M4 = FixedRankMatrices(4, 3, 2)
        p4 = SVDMPoint(
            [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0], [2.0, 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0]
        )
        X4 = UMVTangentVector(
            [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 1.0], [1.0 2.0; 3.0 4.0], [0.0 0.0 1.0; 0.0 0.0 2.0]
        )
        Manifolds.Test.test_manifold(
            M4,
            Dict(
                :Functions => [is_vector, manifold_dimension],
                :Points => [p4], :Vectors => [X4],
                :InvalidVectors => [UMVTangentVector(X4.U, X4.M, p4.Vt)],
            ),
            # Expectations
            Dict(manifold_dimension => 10, :IsVectorErrors => [DomainError]),
        )
    end

    @testset "default retraction and inverse retraction invert each other" begin
        # at a generic point
        M5 = FixedRankMatrices(5, 4, 2)
        p5 = SVDMPoint([cos(1.3i - 0.7j) + 0.1 * i * j for i in 1:5, j in 1:4], 2)
        X5 = project(M5, p5, [sin(0.9i + 1.1j) + 0.2 * (i - j) for i in 1:5, j in 1:4])
        @test isapprox(M5, p5, inverse_retract(M5, p5, retract(M5, p5, X5)), X5)
    end

    @testset "Complex Fixed Rank Matrices" begin
        Mc = FixedRankMatrices(3, 2, 2, ℂ)
        pc = SVDMPoint(ComplexF64[1.0 0.0; 0.0 1.0im; 0.0 0.0], 2)
        Manifolds.Test.test_manifold(
            Mc,
            Dict(
                :Functions => [
                    get_embedding, is_flat, is_point, manifold_dimension, repr,
                    representation_size,
                ],
                :Points => [pc],
            ),
            # Expectations
            Dict(
                get_embedding => Euclidean(3, 2; field = ℂ),
                is_flat => false,
                manifold_dimension => 12,
                repr => "FixedRankMatrices(3, 2, 2, ℂ)",
                representation_size => (3, 2),
            ),
        )
    end

    Manifolds.Test.test_manifold(
        FixedRankMatrices(3, 2, 2; parameter = :field),
        Dict(:Functions => [get_embedding, is_point, manifold_dimension, repr], :Points => [p]),
        # Expectations
        Dict(
            get_embedding => Euclidean(3, 2; parameter = :field),
            manifold_dimension => 6,
            repr => "FixedRankMatrices(3, 2, 2, ℝ; parameter=:field)",
        ),
    )

    @testset "SVDMPoint Basics" begin
        @test p == SVDMPoint(p.U, p.S, p.Vt)
        s = svd(p.U * Diagonal(p.S) * p.Vt)
        x2 = SVDMPoint(s)
        x3 = SVDMPoint(s.U, s.S, s.Vt)
        @test p.S == x2.S
        @test p.U == x2.U
        @test p.Vt == x2.Vt
        @test p == x2
        @test x2.U == x3.U
        @test x2.S == x3.S
        @test x2.Vt == x3.Vt
        @test x2 == x3
        y = SVDMPoint([1.0 0.0; 0.0 0.0; 0.0 0.0], 1)
        s2 = svd([1.0 0.0; 0.0 0.0; 0.0 0.0])
        y2 = SVDMPoint(s2, 1)
        y3 = SVDMPoint(s2.U, s2.S, s2.Vt, 1)
        @test y.S == y2.S
        @test y.U == y2.U
        @test y.Vt == y2.Vt
        @test y == y2
        @test y2.U == y3.U
        @test y2.S == y3.S
        @test y2.Vt == y3.Vt
        @test y2 == y3
        # copyto
        yC = allocate(y)
        copyto!(M, yC, y)
        @test yC.U == y.U
        @test yC.S == y.S
        @test yC.Vt == y.Vt
        pc = copy(p)
        @test pc == p
        @test pc.U !== p.U
        @test is_point(M, pc)
        oneP = SVDMPoint(one(zeros(3, 3)), ones(2), one(zeros(2, 2)), 2)
        @test oneP == one(p)
    end

    @testset "UMVTangentVector Basics" begin
        @test X == UMVTangentVector(X.U, X.M, X.Vt)
        w = UMVTangentVector(X.U, 2 * X.M, X.Vt)
        @test X + w == UMVTangentVector(2 * X.U, 3 * X.M, 2 * X.Vt)
        @test X - w == UMVTangentVector(0 * X.U, -X.M, 0 * X.Vt)
        @test 2 * X == UMVTangentVector(2 * X.U, 2 * X.M, 2 * X.Vt)
        @test X * 2 == UMVTangentVector(X.U * 2, X.M * 2, X.Vt * 2)
        @test 2 \ X == UMVTangentVector(2 \ X.U, 2 \ X.M, 2 \ X.Vt)
        @test X / 2 == UMVTangentVector(X.U / 2, X.M / 2, X.Vt / 2)
        @test +X == X
        @test -X == UMVTangentVector(-X.U, -X.M, -X.Vt)
        w = allocate(X, number_eltype(X))
        zero_vector!(M, w, p)
        @test w == zero_vector(M, p)
        # copyto
        w2 = allocate(w)
        copyto!(M, w2, p, w)
        @test w.U == w2.U
        @test w.M == w2.M
        @test w.Vt == w2.Vt
        # broadcasting
        @test axes(w) === ()
        wc = copy(w)
        # test that the copy is equal to the original, but represented by
        # a new array
        @test wc.U !== w.U
        @test wc.U == w.U
        wb = w .+ X .* 2
        @test wb isa UMVTangentVector
        @test wb == w + X * 2
        @test (wb .= 2 .* w .+ X) == 2 * w + X
        @test wb == 2 * w + X
        wb .= w
        @test wb == w
    end

    @testset "Riemannian Hessian" begin
        Xh = UMVTangentVector(
            [0.0 0.0; 0.0 0.0; 1.0 1.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]
        )
        G = [1.0 0.0; 0.0 2.0; 0.0 0.0]
        H = [0.0 3.0; 0.0 4.0; 0.0 1.0]
        @test is_vector(M, p, riemannian_Hessian(M, p, G, H, Xh))
        M3 = FixedRankMatrices(4, 3, 2)
        p3 = SVDMPoint(
            [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0], [2.0, 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0]
        )
        X3 = UMVTangentVector(
            [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 1.0], [1.0 2.0; 3.0 4.0], [0.0 0.0 1.0; 0.0 0.0 2.0]
        )
        G3 = [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0; 1.0 1.0 1.0]
        H3 = [0.0 3.0 0.0; 0.0 4.0 0.0; 0.0 1.0 0.0; 2.0 0.0 1.0]
        Y3 = riemannian_Hessian(M3, p3, G3, H3, X3)
        @test is_vector(M3, p3, Y3)
        # projection of H3 plus the curvature terms, computed by hand
        @test Y3.U ≈ [0.0 0.0; 0.0 0.0; 1.5 7.0; 2.5 2.0]
        @test Y3.M ≈ [0.0 3.0; 0.0 4.0]
        @test Y3.Vt ≈ [0.0 0.0 1.5; 0.0 0.0 1.0]
    end

    @testset "random points and tangent vectors" begin
        Mc = FixedRankMatrices(3, 2, 2, ℂ)
        pc = rand(MersenneTwister(42), Mc)
        @test number_eltype(pc) === ComplexF64
        @test is_point(Mc, pc)
        @test number_eltype(rand(MersenneTwister(43), Mc; vector_at = pc)) === ComplexF64
        X1 = rand(MersenneTwister(43), M; vector_at = p)
        X2 = rand(MersenneTwister(43), M; vector_at = p, σ = 2.0)
        @test isapprox(M, p, 2 * X1, X2)
        @test rand(MersenneTwister(44), M; σ = 3.0).S == 3 * rand(MersenneTwister(44), M).S
    end

    @testset "Orthographic retraction" begin
        # n = k, so every matrix is tangent and the retraction is p + X
        q1 = retract(M, p, X, OrthographicRetraction())
        @test isapprox(embed(M, q1), [2.0 0.0; 0.0 2.0; 1.0 1.0])
        # n > k, the retraction adds U_X (S + M)^{-1} V_X^T to p + X
        M2 = FixedRankMatrices(4, 3, 2)
        p2 = SVDMPoint(
            [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0], [2.0, 1.0], [1.0 0.0 0.0; 0.0 -1.0 0.0]
        )
        X2 = UMVTangentVector(
            [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 1.0], [0.0 1.0; 0.0 0.0], [0.0 0.0 1.0; 0.0 0.0 0.0]
        )
        q2 = retract(M2, p2, X2, OrthographicRetraction())
        @test isapprox(embed(M2, q2), [2.0 -1.0 1.0; 0.0 -1.0 0.0; 1.0 0.0 0.5; 0.0 -1.0 0.0])
        @test isapprox(M2, p2, inverse_retract(M2, p2, q2, OrthographicInverseRetraction()), X2)
    end

    @testset "complex tangent vectors" begin
        M4c = FixedRankMatrices(4, 3, 2, ℂ)
        p4c = SVDMPoint(
            [1.0im 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0], [2.0, 1.0], [1.0 0.0 0.0; 0.0 1.0im 0.0]
        )
        X4c = UMVTangentVector(
            [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 im], [1.0 2im; 3.0 4.0], [0.0 0.0 im; 0.0 0.0 2.0]
        )
        Y4c = [1.0im -2.0im -1.0; 3.0 4.0im 2.0; 1.0 0.0 0.0; 0.0 -1.0 0.0]
        @test eltype(zero_vector(M4c, p4c).M) === ComplexF64
        @test embed(M4c, p4c, X4c) ≈ Y4c
        @test is_vector(M4c, p4c, X4c)
        @test isapprox(M4c, p4c, project(M4c, p4c, Y4c), X4c)
    end
end
