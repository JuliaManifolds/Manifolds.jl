include("utils.jl")

struct EmptyManifold <: Manifold end
struct EMPoint <: MPoint end
struct ETVector <: TVector end

M = EmptyManifold()
# the EmptyManifold M is not a decorator, so base_manifold returns M
@test base_manifold(M) == M

# test that all functions are now not implemented and result in errors
@test_throws ErrorException manifold_dimension(M)
@test_throws ErrorException representation_size(M)
x = EMPoint()
v = ETVector()
@test_throws ErrorException project_point!(M,x)
@test_throws ErrorException project_tangent!(M,x,1,2)
@test_throws ErrorException inner(M,x,v,v)
@test_throws ErrorException exp!(M,x,v)
@test_throws ErrorException exp!(M,x,v,1.)
@test_throws ErrorException log!(M,x,x)
@test_throws ErrorException vector_transport_along!(M,x,x,v,x)
@test_throws ErrorException hat!(M,v,x,v)
@test_throws ErrorException vee!(M,x,v)
@test_throws ErrorException is_manifold_point(M,x)
@test_throws ErrorException is_manifold_point(M,x,v)
