include("utils.jl")

@testset "Graph Manifold" begin
    M = Euclidean(2)
    x = [1. 2. 3.;4. 5. 6.]
    y = [4. 6. 8.;5. 7. 9.]
    z = [6. 4. 2.;5. 3. 8.]
    @testset "Plain Graph" begin
        G = SimpleGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)
        N = GraphManifold(G,M,VertexManifold())
        @test representation_size(N) == (2,3)
        @test manifold_dimension(N) == manifold_dimension(M)*nv(G)
        @test manifold_dimension(GraphManifold(G,M,EdgeManifold())) == manifold_dimension(M)*ne(G)
        @test is_manifold_point(N,x)
        @test !is_manifold_point(N,x[:,[1:3...,3]]) # an entry too much
        @test_throws DomainError is_manifold_point(N,x[:,[1:3...,3]],true)
        @test is_tangent_vector(N,x,log(N,x,y))
        @test !is_tangent_vector(N,x[:,1:2],log(N,x,y))
        @test_throws DomainError is_tangent_vector(N,x[:,1:2],log(N,x,y),true)
        @test !is_tangent_vector(N,x[:,1:2],log(N,x,y)[:,1:2])
        @test_throws DomainError is_tangent_vector(N,x,log(N,x,y)[:,1:2],true)
        @test incident_log(N,x)==cat(x[:,2]-x[:,1], x[:,1]-x[:,2] + x[:,3]-x[:,2], x[:,2]-x[:,3]; dims=2)

        pts = [x,y,z]
        test_manifold(N, pts)

        NE = GraphManifold(G,M,EdgeManifold())
        @test representation_size(NE) == (2,2)
        @test is_manifold_point(NE,x[:,1:2])
        @test !is_manifold_point(NE,x) # an entry too much
        @test_throws DomainError is_manifold_point(NE,x,true)
        @test is_tangent_vector(NE,x[:,1:2],log(N,x,y)[:,1:2])
        @test !is_tangent_vector(NE,x,log(N,x,y))
        @test_throws DomainError is_tangent_vector(NE,x,log(N,x,y),true)
        @test !is_tangent_vector(N,x[:,1:2],log(N,x,y))
        @test_throws DomainError is_tangent_vector(NE,x[:,1:2],log(N,x,y),true)

        test_manifold(NE, [ x[:,1:2], y[:,1:2], z[:,1:2] ])

        G2 = SimpleDiGraph(3)
        add_edge!(G2, 1, 2)
        add_edge!(G2, 2, 3)
        N2 = GraphManifold(G2,M,VertexManifold())
        @test incident_log(N2,x)==cat(x[:,2]-x[:,1], x[:,3]-x[:,2], [0.;0.]; dims=2)
    end
    @testset "Weighted Graph" begin
        G3 = SimpleWeightedGraph(3)
        add_edge!(G3, 1, 2, 1.5)
        add_edge!(G3, 2, 3, 0.5)
        N3 = GraphManifold(G3,M,VertexManifold())
        @test incident_log(N3,x)==cat(1.5*(x[:,2]-x[:,1]), 1.5*(x[:,1]-x[:,2]) + 0.5*(x[:,3]-x[:,2]), 0.5*(x[:,2]-x[:,3]); dims=2)
    end
end
