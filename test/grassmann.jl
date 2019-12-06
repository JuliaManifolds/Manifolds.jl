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
    end
    types =  [ Matrix{Float32},
            Matrix{Float64}
        ]
    for T in types
        @testset "Type $T" begin
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            y = exp(M,x, [0.0 0.0; 1.0 0.0; 0.0 1.0])
            z = [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2); 0. 0.]
            pts = convert.(T, [x,y,z])
            test_manifold(M,
                          pts,
                          test_project_tangent = true,
                          test_vector_transport = false,
                          test_forward_diff = false,
                          test_reverse_diff = false,
                          retraction_methods = [PolarRetraction(), QRRetraction()],
                          inverse_retraction_methods = [PolarInverseRetraction(), QRInverseRetraction()]
            )
        end
    end
end
