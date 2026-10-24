using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test
using Graphs, SimpleWeightedGraphs

@testset "Graph manifold" begin
    M = Euclidean(2)
    # the path 1 - 2 - 3
    G = SimpleGraph(3)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)
    N = GraphManifold(G, M, VertexManifold())
    NE = GraphManifold(G, M, EdgeManifold())

    x = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    y = [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]
    z = [[6.0, 5.0], [4.0, 3.0], [2.0, 8.0]]
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    Y = [[2.0, 1.0], [0.0, -1.0], [1.0, 0.0]]
    Z = [[0.0, 2.0], [-1.0, 0.0], [0.5, 0.5]]
    B = DefaultOrthonormalBasis()

    Manifolds.Test.test_manifold(
        N,
        Dict(
            :Bases => [B],
            :Coordinates => [vcat(Y...)],
            :Functions => [
                copy, copyto!, distance, exp, get_basis, get_coordinates, get_vector,
                get_vectors, inner, inverse_retract, is_flat, is_point, is_vector, log,
                manifold_dimension, mid_point, norm, parallel_transport_to, retract,
                vector_transport_to, zero_vector,
            ],
            # an entry too much
            :InvalidPoints => [[x..., [0.0, 0.0]]],
            :InvalidVectors => [X[1:2]],
            :InverseRetractionMethods => [LogarithmicInverseRetraction()],
            :Name => "Manifolds.Test suite for the graph manifold on vertices",
            :Points => [x, y, z],
            :RetractionMethods => [ExponentialRetraction()],
            :SecondVector => Y,
            :Vectors => [X, Y, Z],
            :VectorTransportMethods => [ParallelTransport()],
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            distance => 8.0,
            exp => [[2.0, 4.0], [2.0, 6.0], [4.0, 7.0]],
            (get_coordinates, B) => [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            (get_vector, vcat(Y...), B) => Y,
            (get_vectors, B) => :Orthonormal,
            inner => 2.0,
            (inverse_retract, LogarithmicInverseRetraction()) => y - x,
            is_flat => true,
            log => y - x,
            manifold_dimension => 6,
            mid_point => (x + y) / 2,
            norm => 2.0,
            parallel_transport_to => X,
            (retract, ExponentialRetraction()) => x + X,
            (vector_transport_to, ParallelTransport()) => X,
            :IsPointErrors => [DomainError],
            :IsVectorErrors => [DomainError],
            :IsVectorBasepointError => DomainError,
        ),
    )

    # the edges carry the first two entries
    Manifolds.Test.test_manifold(
        NE,
        Dict(
            :Functions => [
                distance, exp, inner, is_flat, is_point, is_vector, log, manifold_dimension,
                norm, zero_vector,
            ],
            :InvalidPoints => [x],
            :InvalidVectors => [X],
            :Name => "Manifolds.Test suite for the graph manifold on edges",
            :Points => [x[1:2], y[1:2]],
            :SecondVector => Y[1:2],
            :Vectors => [X[1:2], Y[1:2]],
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            distance => sqrt(30.0),
            exp => [[2.0, 4.0], [2.0, 6.0]],
            inner => 1.0,
            is_flat => true,
            log => (y - x)[1:2],
            manifold_dimension => 4,
            norm => sqrt(2),
            :IsPointErrors => [DomainError],
            :IsVectorErrors => [DomainError],
            :IsVectorBasepointError => DomainError,
        ),
    )

    @testset "Equality" begin
        @test N == GraphManifold(copy(G), M, VertexManifold())
        @test hash(N) == hash(GraphManifold(copy(G), M, VertexManifold()))
        @test N != NE
        S1 = GraphManifold(G, GeneralizedStiefel(2, 1, [1.0 0.0; 0.0 1.0]), VertexManifold())
        S2 = GraphManifold(G, GeneralizedStiefel(2, 1, [1.0 -0.0; 0.0 1.0]), VertexManifold())
        @test S1 == S2
        @test !isequal(S1, S2)
    end

    @testset "show" begin
        @test sprint(show, "text/plain", N) == """
            GraphManifold
            Graph:
             {3, 2} undirected simple Int64 graph
            AbstractManifold on vertices:
             Euclidean(2; field=ℝ)"""
        @test sprint(show, "text/plain", NE) == """
            GraphManifold
            Graph:
             {3, 2} undirected simple Int64 graph
            AbstractManifold on edges:
             Euclidean(2; field=ℝ)"""
    end

    @testset "Incident logarithms" begin
        # each edge of an undirected graph contributes to both of its vertices
        V = [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]]
        @test incident_log(N, x) == V
        W = zero_vector(N, x)
        @test Manifolds.incident_log!(N, W, x) === W
        @test W == V
        # of a directed graph only to its source
        G2 = SimpleDiGraph(3)
        add_edge!(G2, 1, 2)
        add_edge!(G2, 2, 3)
        @test incident_log(GraphManifold(G2, M, VertexManifold()), x) ==
            [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        # and of a weighted graph scaled by the weights
        G3 = SimpleWeightedGraph(3)
        add_edge!(G3, 1, 2, 1.5)
        add_edge!(G3, 2, 3, 0.5)
        N3 = GraphManifold(G3, M, VertexManifold())
        @test N3 == GraphManifold(copy(G3), M, VertexManifold())
        @test hash(N3) == hash(GraphManifold(copy(G3), M, VertexManifold()))
        V3 = [[1.5, 1.5], [-1.0, -1.0], [-0.5, -0.5]]
        @test incident_log(N3, x) == V3
        W3 = zero_vector(N3, x)
        @test Manifolds.incident_log!(N3, W3, x) === W3
        @test W3 == V3
    end
end
