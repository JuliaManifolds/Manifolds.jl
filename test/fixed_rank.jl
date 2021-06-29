using DoubleFloats: isapprox
include("utils.jl")

@testset "fixed Rank" begin
    M = FixedRankMatrices(3, 2, 2)
    M2 = FixedRankMatrices(3, 2, 1)
    Mc = FixedRankMatrices(3, 2, 2, ℂ)
    x = SVDMPoint([1.0 0.0; 0.0 1.0; 0.0 0.0])
    x2 = SVDMPoint([1.0 0.0; 0.0 1.0; 0.0 0.0], 1)
    v = UMVTVector([0.0 0.0; 0.0 0.0; 1.0 1.0], [1.0 0.0; 0.0 1.0], zeros(2, 2))
    @test repr(M) == "FixedRankMatrices(3, 2, 2, ℝ)"
    @test repr(Mc) == "FixedRankMatrices(3, 2, 2, ℂ)"
    @test sprint(show, "text/plain", x) == """
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
    @test sprint(show, "text/plain", v) == """
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

    @test inner(M, x, v, v) == norm(M, x, v)^2
    @test x == SVDMPoint(x.U, x.S, x.Vt)
    @test v == UMVTVector(v.U, v.M, v.Vt)
    @testset "Fixed Rank Matrices – Basics" begin
        @test representation_size(M) == (3, 2)
        @test get_embedding(M) == Euclidean(3, 2; field=ℝ)
        @test representation_size(Mc) == (3, 2)
        @test manifold_dimension(M) == 6
        @test manifold_dimension(Mc) == 12
        @test !is_point(M, SVDMPoint([1.0 0.0; 0.0 0.0], 2))
        @test_throws DomainError is_point(M, SVDMPoint([1.0 0.0; 0.0 0.0], 2), true)
        @test is_point(M2, x2)

        @test !is_vector(
            M,
            SVDMPoint([1.0 0.0; 0.0 1.0; 0.0 0.0]),
            UMVTVector(zeros(2, 1), zeros(1, 2), zeros(2, 2)),
        )
        @test !is_vector(M, SVDMPoint([1.0 0.0; 0.0 0.0], 2), v)
        @test_throws DomainError is_vector(M, SVDMPoint([1.0 0.0; 0.0 0.0], 2), v, true)
        @test !is_vector(M, x, UMVTVector(x.U, v.M, x.Vt, 2))
        @test_throws DomainError is_vector(M, x, UMVTVector(x.U, v.M, x.Vt, 2), true)
        @test !is_vector(M, x, UMVTVector(v.U, v.M, x.Vt, 2))
        @test_throws DomainError is_vector(M, x, UMVTVector(v.U, v.M, x.Vt, 2), true)

        @test is_point(M, x)
        @test is_vector(M, x, v)
    end
    types = [[Matrix{Float64}, Vector{Float64}, Matrix{Float64}]]
    TEST_FLOAT32 && push!(types, [Matrix{Float32}, Vector{Float32}, Matrix{Float32}])

    for T in types
        @testset "Type $T" begin
            y = retract(M, x, v, PolarRetraction())
            z = SVDMPoint([1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2); 0.0 0.0])
            pts = []
            for p in [x, y, z]
                push!(pts, SVDMPoint(convert.(T, [p.U, p.S, p.Vt])...))
            end
            for p in pts
                @test is_point(M, p)
            end
            @testset "SVD AbstractManifoldPoint Basics" begin
                s = svd(x.U * Diagonal(x.S) * x.Vt)
                x2 = SVDMPoint(s)
                x3 = SVDMPoint(s.U, s.S, s.Vt)
                @test SVDMPoint(x.U, x.S, x.Vt) == x
                @test x.S == x2.S
                @test x.U == x2.U
                @test x.Vt == x2.Vt
                @test x == x2
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

                @test is_point(M, x)
                xM = x.U * Diagonal(x.S) * x.Vt
                @test is_point(M, xM)
                @test !is_point(M, xM[1:2, :])
                @test_throws DomainError is_point(M, xM[1:2, :], true)
                @test_throws DomainError is_point(FixedRankMatrices(3, 2, 1), x, true)
                @test_throws DomainError is_point(FixedRankMatrices(3, 2, 1), xM, true)
                xF1 = SVDMPoint(2 * x.U, x.S, x.Vt)
                @test !is_point(M, xF1)
                @test_throws DomainError is_point(M, xF1, true)
                xF2 = SVDMPoint(x.U, x.S, 2 * x.Vt)
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
                A = embed(M, x)
                @test isapprox(N, A, x.U * Diagonal(x.S) * x.Vt)
            end
            @testset "UMV TVector Basics" begin
                w = UMVTVector(v.U, 2 * v.M, v.Vt)
                @test v + w == UMVTVector(2 * v.U, 3 * v.M, 2 * v.Vt)
                @test v - w == UMVTVector(0 * v.U, -v.M, 0 * v.Vt)
                @test 2 * v == UMVTVector(2 * v.U, 2 * v.M, 2 * v.Vt)
                @test v * 2 == UMVTVector(v.U * 2, v.M * 2, v.Vt * 2)
                @test 2 \ v == UMVTVector(2 \ v.U, 2 \ v.M, 2 \ v.Vt)
                @test v / 2 == UMVTVector(v.U / 2, v.M / 2, v.Vt / 2)
                @test +v == v
                @test -v == UMVTVector(-v.U, -v.M, -v.Vt)
                w = UMVTVector(v.U, v.M, v.Vt)
                @test v == w
                w = allocate(v, number_eltype(v))
                zero_vector!(M, w, x)
                oneP = SVDMPoint(one(zeros(3, 3)), ones(2), one(zeros(2, 2)), 2)
                @test oneP == one(x)
                oneV = UMVTVector(one(zeros(3, 3)), one(zeros(2, 2)), one(zeros(2, 2)), 2)
                @test oneV == one(v)

                # copyto
                w2 = allocate(w)
                copyto!(M, w2, x, w)
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
                wb = w .+ v .* 2
                @test wb isa UMVTVector
                @test wb == w + v * 2
                wb .= 2 .* w .+ v
                @test wb == 2 * w + v
                wb .= w
                @test wb == w
                # embed/project
                N = get_embedding(M)
                B = embed(M, x, v)
                @test isapprox(N, x, B, x.U * v.M * x.Vt + v.U * x.Vt + x.U * v.Vt)
                v2 = project(M, x, B)
                @test isapprox(M, x, v, v2)
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
                test_tangent_vector_broadcasting=false, #broadcast not so easy for 3 matrix type
                projection_atol_multiplier=15,
                retraction_methods=[PolarRetraction()],
                mid_point12=nothing,
            )
        end
    end
end
