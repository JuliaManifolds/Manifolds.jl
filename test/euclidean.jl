include("utils.jl")

@testset "Euclidean" begin
    E = Manifolds.Euclidean(3)
    Ec = Manifolds.Euclidean(3;field=ℂ)
    EM = Manifolds.MetricManifold(E,Manifolds.EuclideanMetric())
    @test is_default_metric(EM) == Val{true}()
    @test is_default_metric(E,Manifolds.EuclideanMetric()) == Val{true}()
    x = zeros(3)
    @test det_local_metric(EM,x) == one(eltype(x))
    @test log_local_metric_density(EM,x) == zero(eltype(x))
    @test project_point!(E,x) == x
    @test manifold_dimension(Ec) == 2*manifold_dimension(E)

    manifolds = [ E, EM, Ec ]
    types = [
        Vector{Float64},
        MVector{3, Float64},
        Vector{Float32},
        Vector{Double64},
    ]

    types_complex = [
        Vector{ComplexF64},
        MVector{3, ComplexF64},
        Vector{ComplexF32},
        Vector{ComplexDF64},
    ]

    for M in manifolds
        basis_types = if M == E
            (ArbitraryOrthonormalBasis(), ProjectedOrthonormalBasis(:svd), DiagonalizingOrthonormalBasis([1.0, 2.0, 3.0]))
        elseif M == Ec
            (ArbitraryOrthonormalBasis(), DiagonalizingOrthonormalBasis([1.0, 2.0, 3.0]))
        else
            ()
        end
        for T in types
            @testset "$M Type $T" begin
                pts = [convert(T, [1.0, 0.0, 0.0]),
                       convert(T, [0.0, 1.0, 0.0]),
                       convert(T, [0.0, 0.0, 1.0])]
                test_manifold(
                    M,
                    pts,
                    test_reverse_diff = isa(T, Vector),
                    test_project_tangent = true,
                    test_musical_isomorphisms = true,
                    test_vector_transport = true,
                    test_mutating_rand = isa(T, Vector),
                    point_distributions = [Manifolds.projected_distribution(M, Distributions.MvNormal(zero(pts[1]), 1.0))],
                    tvector_distributions = [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)],
                    basis_types_vecs = basis_types,
                    basis_types_to_from = basis_types,
                    basis_has_specialized_diagonalizing_get = true
                )
            end
        end
    end
    for T in types_complex
        @testset "Complex Euclidean, type $T" begin
            pts = [convert(T, [1.0im, -1.0im, 1.0]),
                   convert(T, [0.0, 1.0, 1.0im]),
                   convert(T, [0.0, 0.0, 1.0])]
            test_manifold(
                Ec,
                pts,
                test_reverse_diff = isa(T, Vector),
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true
            )
        end
    end

    @testset "hat/vee" begin
        E = Euclidean(3, 2)
        x = collect(reshape(1.0:6.0, (3, 2)))
        v = collect(reshape(7.0:12.0, (3, 2)))
        @test hat(E, x, vec(v)) ≈ v
        w = similar(v)
        @test hat!(E, w, x, vec(v)) === w
        @test w ≈ v
        @test vee(E, x, v) ≈ vec(v)
        w = similar(vec(v))
        @test vee!(E, w, x, v) === w
        @test w ≈ vec(v)
    end
end
