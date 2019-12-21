include("utils.jl")

@testset "Stiefel" begin
    M = Stiefel(3,2)
    Mc = Stiefel(3,2,Complex)
    @testset "Stiefel Basics" begin
        @test representation_size(M) == (3,2)
        @test representation_size(Mc) == (3,2)
        @test manifold_dimension(M) == 3
        @test manifold_dimension(Mc) == 8
        @test !is_manifold_point(M,[1., 0., 0., 0.])
        @test !is_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0., 0., 1., 0.])
    end
    types = [Matrix{Float64},
              MMatrix{3, 2, Float64},
              SizedMatrix{3, 2, Float64},
              Matrix{Float32},
              MMatrix{3, 2, Float32},
              SizedMatrix{3, 2, Float64},
            ]
    for T in types
        @testset "Type $T" begin
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            y = exp(M,x, [0.0 0.0; 0.0 0.0; 1.0 1.0])
            z = [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2); 0. 0.]
            pts = convert.(T, [x,y,z])
            v = inverse_retract(M,x,y,PolarInverseRetraction())
            @test inner(M,x,v,v) == norm(M,x,v)^2
            test_manifold(M,
                          pts,
                          test_exp_log = false,
                          default_inverse_retraction_method = PolarInverseRetraction(),
                          test_log_yields_tangent = false,
                          test_project_tangent = true,
                          test_vector_transport = false,
                          test_forward_diff = false,
                          test_reverse_diff = false,
                          projection_atol_multiplier = 15,
                          retraction_methods = [PolarRetraction(), QRRetraction()],
                          inverse_retraction_methods = [PolarInverseRetraction(), QRInverseRetraction()]
            )
        end
    end
end
