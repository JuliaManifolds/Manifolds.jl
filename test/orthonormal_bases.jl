include("utils.jl")

struct ProjManifold <: Manifold end

ManifoldsBase.inner(::ProjManifold, x, w, v) = dot(w, v)
ManifoldsBase.project_tangent!(S::ProjManifold, w, x, v) = (w .= v .- dot(x, v) .* x)
ManifoldsBase.representation_size(::ProjManifold) = (2,3)
ManifoldsBase.manifold_dimension(::ProjManifold) = 5

@testset "Projected orthonormal basis" begin
    M = ProjManifold()
    x = [sqrt(2)/2 0.0 0.0;
         0.0 sqrt(2)/2 0.0]

    pb = basis(M, x, ProjectedOrthonormalBasis(:svd))
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
end

struct NonManifold <: Manifold end

@testset "ManifoldsBase.jl stuff" begin

    @testset "Errors" begin
        m = NonManifold()
        onb = ArbitraryOrthonormalBasis()

        @test_throws ErrorException get_coordinates(m, [0], [0], onb)
        @test_throws ErrorException get_vector(m, [0], [0], onb)
        @test_throws ErrorException basis(m, [0], onb)
    end

    @testset "basis representation" begin
        M = ManifoldsBase.DefaultManifold(3)
        pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
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
end
