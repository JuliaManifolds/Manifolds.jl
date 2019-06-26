using ManifoldMuseum

using LinearAlgebra
using DoubleFloats
using ForwardDiff
using ReverseDiff
using StaticArrays
using Test

"""
    test_manifold(m::Manifold, pts::AbstractVector;
        test_forward_diff = true,
        retraction_methods = [],
        inverse_retraction_methods = [])

Tests general properties of manifold `m`, given at least three different points
that lie on it (contained in `pts`).

# Arguments
- `test_forward_diff = true`: if true, automatic differentiation using ForwardDiff is
tested.
- `retraction_methods = []`: retraction methods that will be tested.
"""
function test_manifold(M::Manifold, pts::AbstractVector;
    test_forward_diff = true,
    test_reverse_diff = true,
    test_tangent_vector_broadcasting = true,
    retraction_methods = [],
    inverse_retraction_methods = [],
    point_distributions = [],
    tvector_distributions = [])
    # log/exp
    length(pts) ≥ 3 || error("Not enough points (at least three expected)")
    isapprox(M, pts[1], pts[2]) && error("Points 1 and 2 are equal")
    isapprox(M, pts[1], pts[3]) && error("Points 1 and 3 are equal")

    @testset "dimension" begin
        @test isa(manifold_dimension(M), Integer)
        @test manifold_dimension(M) ≥ 0
    end

    @testset "injectivity radius" begin
        @test injectivity_radius(M, pts[1]) > 0
        for rm ∈ retraction_methods
            @test injectivity_radius(M, pts[1], rm) > 0
            @test injectivity_radius(M, pts[1], rm) ≤ injectivity_radius(M, pts[1])
        end
    end

    tv1 = log(M, pts[1], pts[2])

    @testset "is_manifold_point / is_tangent_vector" begin
        for pt ∈ pts
            @test is_manifold_point(M, pt)
        end
        @test is_tangent_vector(M, pts[1], tv1)
    end

    @testset "log/exp tests" begin
        @test isapprox(M, pts[2], exp(M, pts[1], tv1))
        @test isapprox(M, pts[1], exp(M, pts[1], tv1, 0))
        @test isapprox(M, pts[2], exp(M, pts[1], tv1, 1))
        @test is_manifold_point(M, retract(M, pts[1], tv1))
        @test isapprox(M, pts[1], retract(M, pts[1], tv1, 0))
        for retr_method ∈ retraction_methods
            @test is_manifold_point(M, retract(M, pts[1], tv1, retr_method))
            @test isapprox(M, pts[1], retract(M, pts[1], tv1, 0, retr_method))
        end
        new_pt = exp(M, pts[1], tv1)
        retract!(M, new_pt, pts[1], tv1)
        @test is_manifold_point(M, new_pt)
        for x ∈ pts
            @test isapprox(M, zero_tangent_vector(M, x), log(M, x, x); atol = eps(eltype(x)))
            @test isapprox(M, zero_tangent_vector(M, x), inverse_retract(M, x, x); atol = eps(eltype(x)))
            for inv_retr_method ∈ inverse_retraction_methods
                @test isapprox(M, zero_tangent_vector(M, x), inverse_retract(M, x, x, inv_retr_method); atol = eps(eltype(x)))
            end
        end
        zero_tangent_vector!(M, tv1, pts[1])
        @test isapprox(M, pts[1], tv1, zero_tangent_vector(M, pts[1]))
        log!(M, tv1, pts[1], pts[2])
        @test norm(M, pts[1], tv1) ≈ sqrt(inner(M, pts[1], tv1, tv1))

        @test isapprox(M, exp(M, pts[1], tv1, 1), pts[2])
        @test isapprox(M, exp(M, pts[1], tv1, 0), pts[1])

        @test distance(M, pts[1], pts[2]) ≈ norm(M, pts[1], tv1)
    end

    @testset "basic linear algebra in tangent space" begin
        @test isapprox(M, pts[1], 0*tv1, zero_tangent_vector(M, pts[1]))
        @test isapprox(M, pts[1], 2*tv1, tv1+tv1)
        @test isapprox(M, pts[1], 0*tv1, tv1-tv1)
        @test isapprox(M, pts[1], (-1)*tv1, -tv1)
    end

    test_tangent_vector_broadcasting && @testset "broadcasted linear algebra in tangent space" begin
        @test isapprox(M, pts[1], 3*tv1, 2 .* tv1 .+ tv1)
        @test isapprox(M, pts[1], -tv1, tv1 .- 2 .* tv1)
        @test isapprox(M, pts[1], -tv1, .-tv1)
        v = similar(tv1)
        v .= 2 .* tv1 .+ tv1
        @test v ≈ 3*tv1
    end

    test_forward_diff && @testset "ForwardDiff support" begin
        exp_f(t) = distance(M, pts[1], exp(M, pts[1], t*tv1))
        d12 = distance(M, pts[1], pts[2])
        for t ∈ 0.1:0.1:1.0
            @test d12 ≈ ForwardDiff.derivative(exp_f, t)
        end

        retract_f(t) = distance(M, pts[1], retract(M, pts[1], t*tv1))
        for t ∈ 0.1:0.1:1.0
            @test d12 ≈ ForwardDiff.derivative(retract_f, t)
        end
    end

    test_reverse_diff && @testset "ReverseDiff support" begin
        exp_f(t) = distance(M, pts[1], exp(M, pts[1], t[1]*tv1))
        d12 = distance(M, pts[1], pts[2])
        for t ∈ 0.1:0.1:1.0
            @test d12 ≈ ReverseDiff.gradient(exp_f, [t])[1]
        end

        retract_f(t) = distance(M, pts[1], retract(M, pts[1], t[1]*tv1))
        for t ∈ 0.1:0.1:1.0
            @test d12 ≈ ReverseDiff.gradient(retract_f, [t])[1]
        end
    end

    @testset "eltype" begin
        tv1 = log(M, pts[1], pts[2])
        @test eltype(tv1) == eltype(pts[1])
        @test eltype(exp(M, pts[1], tv1)) == eltype(pts[1])
    end

    @testset "point distributions" begin
        for pd ∈ point_distributions
            for _ in 1:10
                @test is_manifold_point(M, rand(pd))
            end
        end
    end

    @testset "tangent vector distributions" begin
        for tvd ∈ tvector_distributions
            supp = ManifoldMuseum.support(tvd)
            for _ in 1:10
                @test is_tangent_vector(M, supp.x, rand(tvd))
            end
        end
    end
end

function test_arraymanifold()
    M = ManifoldMuseum.Sphere(2)
    A = ArrayManifold(M)
    x = [1., 0., 0.]
    y = 1/sqrt(2)*[1., 1., 0.]
    z = [0., 1., 0.]
    v = log(M,x,y)
    v2 = log(A,x,y)
    y2 = exp(A,x,v2)
    w = log(M,x,z)
    w2 = log(A,x,z; atol=10^(-15))
    @test isapprox(y2.value,y)
    @test distance(A,x,y) == distance(M,x,y)
    @test norm(A,x,v) == norm(M,x,v)
    @test inner(A,x,v2,w2; atol=10^(-15)) == inner(M,x,v,w)
    @test_throws DomainError ManifoldMuseum.is_manifold_point(M,2*y)
    @test_throws DomainError ManifoldMuseum.is_tangent_vector(M,y,v; atol=10^(-15))

    test_manifold(A, [x, y, z],
        test_tangent_vector_broadcasting = false)
end

@testset "Sphere" begin
    M = ManifoldMuseum.Sphere(2)
    types = [Vector{Float64},
             SizedVector{3, Float64},
             MVector{3, Float64},
             Vector{Float32},
             SizedVector{3, Float32},
             MVector{3, Float32},
             Vector{Double64},
             MVector{3, Double64},
             SizedVector{3, Double64}]
    for T in types
        @testset "Type $T" begin
            pts = [convert(T, [1.0, 0.0, 0.0]),
                   convert(T, [0.0, 1.0, 0.0]),
                   convert(T, [0.0, 0.0, 1.0])]
            test_manifold(M,
                          pts,
                          test_reverse_diff = isa(T, Vector),
                          point_distributions = [ManifoldMuseum.uniform_distribution(M, pts[1])],
                          tvector_distributions = [ManifoldMuseum.normal_tvector_distribution(M, pts[1], 1.0)])
        end
    end

    @testset "Distribution tests" begin
        usd_mvector = ManifoldMuseum.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
        @test isa(rand(usd_mvector), MVector)

        gtsd_mvector = ManifoldMuseum.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
        @test isa(rand(gtsd_mvector), MVector)
    end

    test_arraymanifold()
end

@testset "Rotations" begin
    M = ManifoldMuseum.Rotations(2)

    types = [Matrix{Float64},
             SizedMatrix{2, 2, Float64},
             MMatrix{2, 2, Float64},
             Matrix{Float32},
             SizedMatrix{2, 2, Float32},
             MMatrix{2, 2, Float32}]

    retraction_methods = [ManifoldMuseum.PolarRetraction(),
                          ManifoldMuseum.QRRetraction()]

    inverse_retraction_methods = [ManifoldMuseum.PolarInverseRetraction(),
                                  ManifoldMuseum.QRInverseRetraction()]

    for T in types
        angles = (0.0, π/2, 2π/3, π/4)
        pts = [convert(T, [cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]) for ϕ in angles]
        test_manifold(M, pts;
            test_forward_diff = false,
            test_reverse_diff = false,
            retraction_methods = retraction_methods,
            inverse_retraction_methods = inverse_retraction_methods,
            point_distributions = [ManifoldMuseum.normal_rotation_distribution(M, pts[1], 1.0)],
            tvector_distributions = [ManifoldMuseum.normal_tvector_distribution(M, pts[1], 1.0)])

        v = log(M, pts[1], pts[2])
        @test norm(M, pts[1], v) ≈ (angles[2] - angles[1])*sqrt(2)

        v14_polar = inverse_retract(M, pts[1], pts[4], ManifoldMuseum.PolarInverseRetraction())
        p4_polar = retract(M, pts[1], v14_polar, ManifoldMuseum.PolarRetraction())
        @test isapprox(M, pts[4], p4_polar)

        v14_qr = inverse_retract(M, pts[1], pts[4], ManifoldMuseum.QRInverseRetraction())
        p4_qr = retract(M, pts[1], v14_qr, ManifoldMuseum.QRRetraction())
        @test isapprox(M, pts[4], p4_qr)
    end

    @testset "Distribution tests" begin
        usd_mmatrix = ManifoldMuseum.normal_rotation_distribution(M, (@MMatrix [1.0 0.0; 0.0 1.0]), 1.0)
        @test isa(rand(usd_mmatrix), MMatrix)

        gtsd_mvector = ManifoldMuseum.normal_tvector_distribution(M, (@MMatrix [1.0 0.0; 0.0 1.0]), 1.0)
        @test isa(rand(gtsd_mvector), MMatrix)
    end
end
