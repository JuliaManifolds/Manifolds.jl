# Testing

Documentation for testing utilities for `Manifolds.jl`. The function [`test_manifold`](@ref Manifolds.test_manifold)
can be used to verify that your manifold correctly implements the `Manifolds.jl`
interface. Similarly [`test_group`](@ref Manifolds.test_group) and [`test_action`](@ref Manifolds.test_action) can be used to verify implementation of groups and group actions.

```@autodocs
Modules = [Manifolds]
Pages = ["tests/reversediff.jl", "tests/test_forwarddiff.jl","tests/tests_general.jl", "tests/tests_group.jl"]
Order = [:type, :function]
```
