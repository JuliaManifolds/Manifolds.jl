include("../utils.jl")

@testset "fixed Rank" begin
    M = FixedRankMatrices(3, 2, 2)
    M2 = FixedRankMatrices(3, 2, 1)
    Mc = FixedRankMatrices(3, 2, 2, ℂ)
    pE = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    p = SVDMPoint(pE)
    p2 = SVDMPoint([1.0 0.0; 0.0 1.0; 0.0 0.0], 1)
    X = UMVTVector([0.0 0.0; 0.0 0.0; 1.0 1.0], [1.0 0.0; 0.0 1.0], zeros(2, 2))
    @test repr(M) == "FixedRankMatrices(3, 2, 2, ℝ)"
    @test repr(Mc) == "FixedRankMatrices(3, 2, 2, ℂ)"
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
    $(sprint(show, UMVTVector{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}))
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

    @test inner(M, p, X, X) == norm(M, p, X)^2
    @test p == SVDMPoint(p.U, p.S, p.Vt)
    @test X == UMVTVector(X.U, X.M, X.Vt)
    @testset "Fixed Rank Matrices – Basics" begin
        @test representation_size(M) == (3, 2)
        @test get_embedding(M) == Euclidean(3, 2; field=ℝ)
        @test representation_size(Mc) == (3, 2)
        @test manifold_dimension(M) == 6
        @test manifold_dimension(Mc) == 12
        @test !is_point(M, SVDMPoint([1.0 0.0; 0.0 0.0], 2))
        @test_throws DomainError is_point(M, SVDMPoint([1.0 0.0; 0.0 0.0], 2), true)
        @test is_point(M2, p2)
        @test_throws DomainError is_point(M2, [1.0 0.0; 0.0 1.0; 0.0 0.0], true)
        @test Manifolds.check_point(M2, [1.0 0.0; 0.0 1.0; 0.0 0.0]) isa DomainError

        @test !is_vector(
            M,
            SVDMPoint([1.0 0.0; 0.0 1.0; 0.0 0.0]),
            UMVTVector(zeros(2, 1), zeros(1, 2), zeros(2, 2)),
        )
        @test !is_vector(M, SVDMPoint([1.0 0.0; 0.0 0.0], 2), X)
        @test_throws ManifoldDomainError is_vector(
            M,
            SVDMPoint([1.0 0.0; 0.0 0.0], 2),
            X,
            true,
        )
        @test !is_vector(M, p, UMVTVector(p.U, X.M, p.Vt, 2))
        @test_throws DomainError is_vector(M, p, UMVTVector(p.U, X.M, p.Vt, 2), true)
        @test !is_vector(M, p, UMVTVector(X.U, X.M, p.Vt, 2))
        @test_throws DomainError is_vector(M, p, UMVTVector(X.U, X.M, p.Vt, 2), true)

        @test is_point(M, p)
        @test is_vector(M, p, X)

        q = embed(M, p)
        @test pE == q
        q2 = similar(q)
        embed!(M, q2, p)
        @test q == q2
    end
    types = [[Matrix{Float64}, Vector{Float64}, Matrix{Float64}]]
    TEST_FLOAT32 && push!(types, [Matrix{Float32}, Vector{Float32}, Matrix{Float32}])

    for T in types
        @testset "Type $T" begin
            p2 = retract(M, p, X, PolarRetraction())
            p3 = SVDMPoint([1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2); 0.0 0.0])
            pts = []
            for p in [p, p2, p3]
                push!(pts, SVDMPoint(convert.(T, [p.U, p.S, p.Vt])...))
            end
            for p in pts
                @test is_point(M, p)
            end
            @testset "SVD AbstractManifoldPoint Basics" begin
                s = svd(p.U * Diagonal(p.S) * p.Vt)
                x2 = SVDMPoint(s)
                x3 = SVDMPoint(s.U, s.S, s.Vt)
                @test SVDMPoint(p.U, p.S, p.Vt) == p
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

                @test is_point(M, p)
                xM = embed(M, p)
                @test is_point(M, xM)
                @test !is_point(M, xM[1:2, :])
                @test_throws DomainError is_point(M, xM[1:2, :], true)
                @test_throws DomainError is_point(FixedRankMatrices(3, 2, 1), p, true)
                @test_throws DomainError is_point(FixedRankMatrices(3, 2, 1), xM, true)
                xF1 = SVDMPoint(2 * p.U, p.S, p.Vt)
                @test !is_point(M, xF1)
                @test_throws DomainError is_point(M, xF1, true)
                xF2 = SVDMPoint(p.U, p.S, 2 * p.Vt)
                @test !is_point(M, xF2)
                @test_throws DomainError is_point(M, xF2, true)
                # copyto
                yC = allocate(y)
                copyto!(M, yC, y)
                @test yC.U == y.U
                @test yC.S == y.S
                @test yC.Vt == y.Vt
                # embed
                N = get_embedding(M)
                A = embed(M, p)
                @test isapprox(N, A, p.U * Diagonal(p.S) * p.Vt)
            end
            @testset "UMV TVector Basics" begin
                w = UMVTVector(X.U, 2 * X.M, X.Vt)
                @test X + w == UMVTVector(2 * X.U, 3 * X.M, 2 * X.Vt)
                @test X - w == UMVTVector(0 * X.U, -X.M, 0 * X.Vt)
                @test 2 * X == UMVTVector(2 * X.U, 2 * X.M, 2 * X.Vt)
                @test X * 2 == UMVTVector(X.U * 2, X.M * 2, X.Vt * 2)
                @test 2 \ X == UMVTVector(2 \ X.U, 2 \ X.M, 2 \ X.Vt)
                @test X / 2 == UMVTVector(X.U / 2, X.M / 2, X.Vt / 2)
                @test +X == X
                @test -X == UMVTVector(-X.U, -X.M, -X.Vt)
                w = UMVTVector(X.U, X.M, X.Vt)
                @test X == w
                w = allocate(X, number_eltype(X))
                zero_vector!(M, w, p)
                oneP = SVDMPoint(one(zeros(3, 3)), ones(2), one(zeros(2, 2)), 2)
                @test oneP == one(p)

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
                @test wb isa UMVTVector
                @test wb == w + X * 2
                @test (wb .= 2 .* w .+ X) == 2 * w + X
                @test wb == 2 * w + X
                wb .= w
                @test wb == w
                # embed/project
                N = get_embedding(M)
                B = embed(M, p, X)
                @test isapprox(N, p, B, p.U * X.M * p.Vt + X.U * p.Vt + p.U * X.Vt)
                BB = similar(B)
                embed!(M, BB, p, X)
                @test isapprox(M, p, B, BB)
                v2 = project(M, p, B)
                @test isapprox(M, p, X, v2)
            end
            @testset "Projection Retraction and Vector transport" begin
                X2 = vector_transport_to(M, p, X, p2, ProjectionTransport())
                X2t = project(M, p2, embed(M, p, X))
                @test isapprox(M, p, X2, X2t)
            end
            test_manifold(
                M,
                pts,
                test_exp_log=false,
                default_inverse_retraction_method=nothing,
                test_injectivity_radius=false,
                default_retraction_method=PolarRetraction(),
                test_is_tangent=false,
                test_default_vector_transport=false,
                test_forward_diff=false,
                test_reverse_diff=false,
                test_vector_spaces=false,
                test_vee_hat=false,
                test_tangent_vector_broadcasting=true,
                projection_atol_multiplier=15,
                retraction_methods=[PolarRetraction()],
                vector_transport_methods=[ProjectionTransport()],
                vector_transport_retractions=[PolarRetraction()],
                vector_transport_inverse_retractions=[PolarInverseRetraction()],
                mid_point12=nothing,
                test_inplace=true,
                test_rand_point=true,
                test_rand_tvector=true,
            )
        end
    end
end
