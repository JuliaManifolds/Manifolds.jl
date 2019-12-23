include("utils.jl")

@testset "Grassmann" begin
    M = Grassmann(3,2)
    Mc = Grassmann(3,2,Complex)
    @testset "Grassmann Basics" begin
        @test representation_size(M) == (3,2)
        @test representation_size(Mc) == (3,2)
        @test manifold_dimension(M) == 2
        @test manifold_dimension(Mc) == 4
        @test !is_manifold_point(M,[1., 0., 0., 0.])
        @test !is_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0., 0., 1., 0.])
        @test_throws DomainError is_manifold_point(Grassmann(3,2), [2. 0.; 0. 1.; 0. 0.],true)
        @test_throws DomainError is_tangent_vector(Grassmann(3,2), [2. 0.; 0. 1.; 0. 0.],zeros(3,2),true)
        @test_throws DomainError is_tangent_vector(Grassmann(3,2), [1. 0.; 0. 1.; 0. 0.],ones(3,2),true)
    end
    types = [Matrix{Float64},
             MMatrix{3, 2, Float64},
             Matrix{Float32},
             MMatrix{3, 2, Float32}]
    for T in types
        @testset "Type $T" begin
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            v = [0.0 0.0; 0.0 0.0; 0.0 1.0]
            y = exp(M,x,v)
            w = [0.0 1.0; -1.0 0.0; 1.0 0.0]
            z = exp(M,x,w)
            pts = convert.(T, [x,y,z])
            test_manifold(
                M,
                pts,
                test_exp_log = true,
                test_injectivity_radius = false,
                test_project_tangent = true,
                test_vector_transport = false,
                test_forward_diff = false,
                test_reverse_diff = false,
                retraction_methods = [PolarRetraction(), QRRetraction()],
                inverse_retraction_methods = [PolarInverseRetraction(), QRInverseRetraction()],
                exp_log_atol_multiplier = 10.0,
                is_tangent_atol_multiplier = 10.0,
            )
        end
    end
end
