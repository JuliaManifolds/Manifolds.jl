include("utils.jl")

struct ProjManifold <: Manifold end

ManifoldsBase.inner(::ProjManifold, x, w, v) = dot(w, v)
ManifoldsBase.project_tangent!(S::ProjManifold, w, x, v) = (w .= v .- dot(x, v) .* x)
ManifoldsBase.representation_size(::ProjManifold) = (2,3)
ManifoldsBase.manifold_dimension(::ProjManifold) = 5
Manifolds.get_vector(::ProjManifold, x, v, ::ArbitraryOrthonormalBasis) = reverse(v)

@testset "Projected and arbitrary orthonormal basis" begin
    M = ProjManifold()
    x = [sqrt(2)/2 0.0 0.0;
         0.0 sqrt(2)/2 0.0]

    pb = basis(M, x, ProjectedOrthonormalBasis(:svd))
    @test number_system(pb) == ℝ
    @test basis(M, x, pb) == pb
    N = manifold_dimension(M)
    @test isa(pb, PrecomputedOrthonormalBasis)
    @test length(pb.vectors) == N
    # test orthonormality
    for i in 1:N
        @test norm(M, x, pb.vectors[i]) ≈ 1
        for j in i+1:N
            @test inner(M, x, pb.vectors[i], pb.vectors[j]) ≈ 0 atol = 1e-15
        end
    end
    # check projection idempotency
    for i in 1:N
        @test project_tangent(M, x, pb.vectors[i]) ≈ pb.vectors[i]
    end

    aonb = basis(M, x, ArbitraryOrthonormalBasis())
    @test size(aonb.vectors) == (5,)
    @test aonb.vectors[1] ≈ [0, 0, 0, 0, 1]
end

struct NonManifold <: Manifold end
struct NonBasis <: Manifolds.AbstractBasis{ℝ} end

@testset "ManifoldsBase.jl stuff" begin

    @testset "Errors" begin
        m = NonManifold()
        onb = ArbitraryOrthonormalBasis()

        @test_throws ErrorException basis(m, [0], onb)
        @test_throws ErrorException basis(m, [0], NonBasis())
        @test_throws ErrorException get_coordinates(m, [0], [0], onb)
        @test_throws ErrorException get_vector(m, [0], [0], onb)
        @test_throws ErrorException vectors(m, [0], NonBasis())
    end

    M = ManifoldsBase.DefaultManifold(3)
    pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    @testset "basis representation" begin
        v1 = log(M, pts[1], pts[2])

        vb = get_coordinates(M, pts[1], v1, ArbitraryOrthonormalBasis())
        @test isa(vb, AbstractVector)
        vbi = get_vector(M, pts[1], vb, ArbitraryOrthonormalBasis())
        @test isapprox(M, pts[1], v1, vbi)

        b = basis(M, pts[1], ArbitraryOrthonormalBasis())
        @test isa(b, PrecomputedOrthonormalBasis)
        N = manifold_dimension(M)
        @test length(b.vectors) == N
        # check orthonormality
        for i in 1:N
            @test norm(M, pts[1], b.vectors[i]) ≈ 1
            for j in i+1:N
                @test inner(M, pts[1], b.vectors[i], b.vectors[j]) ≈ 0
            end
        end
        # check that the coefficients correspond to the basis
        for i in 1:N
            @test inner(M, pts[1], v1, b.vectors[i]) ≈ vb[i]
        end

        @test get_coordinates(M, pts[1], v1, b) ≈ get_coordinates(M, pts[1], v1, ArbitraryOrthonormalBasis())
        @test get_vector(M, pts[1], vb, b) ≈ get_vector(M, pts[1], vb, ArbitraryOrthonormalBasis())

    end

    @testset "ArrayManifold basis" begin
        A = ArrayManifold(M)
        aonb = ArbitraryOrthonormalBasis()
        b = basis(A, pts[1], aonb)
        @test_throws ErrorException get_vector(A, pts[1], [], aonb)
        @test_throws DimensionMismatch get_coordinates(A, pts[1], [], aonb)
        @test_throws ArgumentError basis(A, pts[1], PrecomputedOrthonormalBasis([pts[1]]))
        @test_throws ArgumentError basis(A, pts[1], PrecomputedOrthonormalBasis([pts[1], pts[1], pts[1]]))
        @test_throws ArgumentError basis(A, pts[1], PrecomputedOrthonormalBasis([2*pts[1], pts[1], pts[1]]))
    end
end
