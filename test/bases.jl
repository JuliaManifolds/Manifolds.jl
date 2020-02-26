include("utils.jl")

struct ProjManifold <: Manifold end

ManifoldsBase.inner(::ProjManifold, x, w, v) = dot(w, v)
ManifoldsBase.project_tangent!(S::ProjManifold, w, x, v) = (w .= v .- dot(x, v) .* x)
ManifoldsBase.representation_size(::ProjManifold) = (2,3)
ManifoldsBase.manifold_dimension(::ProjManifold) = 5
Manifolds.get_vector(::ProjManifold, x, v, ::DefaultOrthonormalBasis) = reverse(v)

@testset "Projected and arbitrary orthonormal basis" begin
    M = ProjManifold()
    x = [sqrt(2)/2 0.0 0.0;
         0.0 sqrt(2)/2 0.0]

    pb = get_basis(M, x, ProjectedOrthonormalBasis(:svd))
    @test number_system(pb) == ℝ
    @test get_basis(M, x, pb) == pb
    N = manifold_dimension(M)
    @test isa(pb, CachedBasis)
    @test length(get_vectors(M, x, pb)) == N
    # test orthonormality
    for i in 1:N
        @test norm(M, x, get_vectors(M, x, pb)[i]) ≈ 1
        for j in i+1:N
            @test inner(M, x, get_vectors(M, x, pb)[i], get_vectors(M, x, pb)[j]) ≈ 0 atol = 1e-15
        end
    end
    # check projection idempotency
    for i in 1:N
        @test project_tangent(M, x, get_vectors(M, x, pb)[i]) ≈ get_vectors(M, x, pb)[i]
    end

    aonb = get_basis(M, x, DefaultOrthonormalBasis())
    @test size(get_vectors(M, x, aonb)) == (5,)
    @test get_vectors(M, x, aonb)[1] ≈ [0, 0, 0, 0, 1]
end

struct NonManifold <: Manifold end
struct NonBasis <: Manifolds.AbstractBasis{ℝ} end

@testset "ManifoldsBase.jl stuff" begin

    @testset "Errors" begin
        m = NonManifold()
        onb = DefaultOrthonormalBasis()

        @test_throws ErrorException get_basis(m, [0], onb)
        @test_throws ErrorException get_basis(m, [0], NonBasis())
        @test_throws ErrorException get_coordinates(m, [0], [0], onb)
        @test_throws ErrorException get_coordinates!(m, [0], [0], [0], onb)
        @test_throws ErrorException get_vector(m, [0], [0], onb)
        @test_throws ErrorException get_vector!(m, [0], [0], [0], onb)
        @test_throws ErrorException get_vectors(m, [0], NonBasis())
    end

    M = ManifoldsBase.DefaultManifold(3)
    pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    @testset "basis representation" begin
        v1 = log(M, pts[1], pts[2])

        vb = get_coordinates(M, pts[1], v1, DefaultOrthonormalBasis())
        @test isa(vb, AbstractVector)
        vbi = get_vector(M, pts[1], vb, DefaultOrthonormalBasis())
        @test isapprox(M, pts[1], v1, vbi)

        b = get_basis(M, pts[1], DefaultOrthonormalBasis())
        @test isa(b, CachedBasis{DefaultOrthonormalBasis{ℝ},Array{Array{Float64,1},1},ℝ})
        N = manifold_dimension(M)
        @test length(get_vectors(M, pts[1], b)) == N
        # check orthonormality
        for i in 1:N
            @test norm(M, pts[1], get_vectors(M, pts[1], b)[i]) ≈ 1
            for j in i+1:N
                @test inner(
                    M,
                    pts[1],
                    get_vectors(M, pts[1], b)[i],
                    get_vectors(M, pts[1], b)[j]
                ) ≈ 0
            end
        end
        # check that the coefficients correspond to the basis
        for i in 1:N
            @test inner(M, pts[1], v1, get_vectors(M, pts[1], b)[i]) ≈ vb[i]
        end

        @test get_coordinates(M, pts[1], v1, b) ≈ get_coordinates(M, pts[1], v1, DefaultOrthonormalBasis())
        @test get_vector(M, pts[1], vb, b) ≈ get_vector(M, pts[1], vb, DefaultOrthonormalBasis())

        v1c = allocate(v1)
        get_coordinates!(M, v1c, pts[1], v1, b)
        @test v1c ≈ get_coordinates(M, pts[1], v1, b)

        v1cv = allocate(v1)
        get_vector!(M, v1cv, pts[1], v1c, b)
        @test isapprox(M, pts[1], v1, v1cv)
    end

    @testset "ArrayManifold basis" begin
        A = ArrayManifold(M)
        aonb = DefaultOrthonormalBasis()
        b = get_basis(A, pts[1], aonb)
        @test_throws ErrorException get_vector(A, pts[1], [], aonb)
        @test_throws ArgumentError get_basis(A, pts[1], CachedBasis(aonb,[pts[1]]))
        @test_throws ArgumentError get_basis(A, pts[1], CachedBasis(aonb,[pts[1], pts[1], pts[1]]))
        @test_throws ArgumentError get_basis(A, pts[1], CachedBasis(aonb,[2*pts[1], pts[1], pts[1]]))
    end
end

@testset "Basis show methods" begin
    @test sprint(show, DefaultOrthonormalBasis()) == "DefaultOrthonormalBasis(ℝ)"
    @test sprint(show, DefaultOrthonormalBasis(ℂ)) == "DefaultOrthonormalBasis(ℂ)"
    @test sprint(show, ProjectedOrthonormalBasis(:svd)) == "ProjectedOrthonormalBasis(:svd, ℝ)"
    @test sprint(show, ProjectedOrthonormalBasis(:gram_schmidt, ℂ)) == "ProjectedOrthonormalBasis(:gram_schmidt, ℂ)"

    @test sprint(show, "text/plain", DiagonalizingOrthonormalBasis(Float64[1, 2, 3])) == """
    DiagonalizingOrthonormalBasis with coordinates in ℝ and eigenvalue 0 in direction:
    3-element Array{Float64,1}:
      1.0
      2.0
      3.0"""

    M = Euclidean(2, 3)
    x = collect(reshape(1.0:6.0, (2, 3)))
    pb = get_basis(M, x, DefaultOrthonormalBasis())
    @test sprint(show, "text/plain", pb) == """
    DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 6 basis vectors:
     E1 =
      2×3 Array{Float64,2}:
       1.0  0.0  0.0
       0.0  0.0  0.0
     E2 =
      2×3 Array{Float64,2}:
       0.0  0.0  0.0
       1.0  0.0  0.0
     ⋮
     E5 =
      2×3 Array{Float64,2}:
       0.0  0.0  1.0
       0.0  0.0  0.0
     E6 =
      2×3 Array{Float64,2}:
       0.0  0.0  0.0
       0.0  0.0  1.0"""
    b = DiagonalizingOrthonormalBasis(get_vectors(M, x, pb)[1])
    dpb = CachedBasis(b, Float64[1, 2, 3, 4, 5, 6], get_vectors(M, x, pb))
    @test sprint(show, "text/plain", dpb) == """
    DiagonalizingOrthonormalBasis with coordinates in ℝ and eigenvalue 0 in direction:
     2×3 Array{Float64,2}:
       1.0  0.0  0.0
       0.0  0.0  0.0
    and 6 basis vectors.
    Basis vectors:
     E1 =
      2×3 Array{Float64,2}:
       1.0  0.0  0.0
       0.0  0.0  0.0
     E2 =
      2×3 Array{Float64,2}:
       0.0  0.0  0.0
       1.0  0.0  0.0
     ⋮
     E5 =
      2×3 Array{Float64,2}:
       0.0  0.0  1.0
       0.0  0.0  0.0
     E6 =
      2×3 Array{Float64,2}:
       0.0  0.0  0.0
       0.0  0.0  1.0
    Eigenvalues:
     6-element Array{Float64,1}:
      1.0
      2.0
      3.0
      4.0
      5.0
      6.0"""

    M = Euclidean(1, 1, 1)
    x = reshape(Float64[1], (1, 1, 1))
    pb = get_basis(M, x, DefaultOrthonormalBasis())
    @test sprint(show, "text/plain", pb) == """
    DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 1 basis vector:
     E1 =
      1×1×1 Array{Float64,3}:
      [:, :, 1] =
       1.0"""

    dpb = CachedBasis(DiagonalizingOrthonormalBasis(get_vectors(M, x, pb)), Float64[1], get_vectors(M, x, pb))
    @test sprint(show, "text/plain", dpb) == """
    DiagonalizingOrthonormalBasis with coordinates in ℝ and eigenvalue 0 in direction:
     1-element Array{Array{Float64,3},1}:
       [1.0]
    and 1 basis vector.
    Basis vectors:
     E1 =
      1×1×1 Array{Float64,3}:
      [:, :, 1] =
       1.0
    Eigenvalues:
     1-element Array{Float64,1}:
      1.0"""
end
