struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: AbstractManifold{ℝ} end

struct NotImplementedGroupDecorator{M} <: AbstractDecoratorManifold{ℝ}
    manifold::M
end
function active_traits(::NotImplementedGroupDecorator)
    return merge_traits(IsEmbeddedSubmanifold())
end

struct DefaultTransparencyGroup{M,A<:AbstractGroupOperation} <: AbstractGroupManifold{ℝ,A}
    manifold::M
    op::A
end
active_traits(::DefaultTransparencyGroup) = merge_traits(IsEmbeddedManifold())
