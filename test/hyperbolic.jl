include("utils.jl")

@testset "Hyperbolic Space" begin
    M = Hyperbolic(2)
    @testset "Hyperbolic Basics" begin
        @test repr(M) == "Hyperbolic(2)"
        @test representation_size(M) == (3,)
        @test !is_manifold_point(M, [1., 0., 0., 0.])
        @test !is_tangent_vector(M, [0.,0.,1.], [0., 0., 1., 0.])
        @test_throws DomainError is_manifold_point(M, [2.,0.,0.],true)
        @test !is_manifold_point(M, [2.,0.,0.])
        @test !is_tangent_vector(M,[1.,0.,0.],[1.,0.,0.])
        @test_throws DomainError is_tangent_vector(M,[1.,0.,0.],[1.,0.,0.],true)
        @test !is_tangent_vector(M,[0.,0.,1.],[1.,0.,1.])
        @test_throws DomainError is_tangent_vector(M,[0.,0.,1.],[1.,0.,1.],true)
        @test is_default_metric(M,MinkowskiMetric()) === Val(true)
        @test manifold_dimension(M) == 2
    end
    types = [
        Vector{Float64},
        SizedVector{3, Float64},
        Vector{Float32},
    ]
    for T in types
        @testset "Type $T" begin
            pts = [convert(T, [0.0, 0.0, 1.0]),
                   convert(T, [1.0, 0.0, sqrt(2.0)]),
                   convert(T, [0.0, 1.0, sqrt(2.0)])]
            test_manifold(
                M,
                pts,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                is_tangent_atol_multiplier = 10.0,
                exp_log_atol_multiplier = 10.0,
            )
        end
    end
    @testset "Hyperbolic mean test" begin
        pts =[
            [0., 0., 1.],
            [1., 0., sqrt(2.0)],
            [-1., 0., sqrt(2.0)],
            [0., 1., sqrt(2.0)],
            [0., -1., sqrt(2.0)]
         ]
         @test isapprox(M,mean(M,pts),pts[1]; atol=10^-4)
    end
end
