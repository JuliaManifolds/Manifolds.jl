include("utils.jl")

@info "Manifolds.jl Test settings:\n\n" *
      "Testing Float32:  $(TEST_FLOAT32)\n" *
      "Testing Double64: $(TEST_DOUBLE64)\n" *
      "Testing Static:   $(TEST_STATIC_SIZED)\n\n" *
      "Test group:       $(TEST_GROUP)\n\n" *
      "These settings are stored in environment variables, see in test/utils.jl"

@testset "Manifolds.jl" begin
    if TEST_GROUP ∈ ["all", "test_manifolds"]
        include_test("manifolds/stiefel.jl")
    end
end
