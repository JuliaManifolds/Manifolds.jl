struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: AbstractManifold{ℝ} end

struct NotImplementedGroupDecorator{M} <:
       AbstractDecoratorManifold{ℝ}
    manifold::M
end
activate_traits(::NotImplementedGroupDecorator) = merge_traits(IsEmbeddedSubmanifoldManifold())

struct DefaultTransparencyGroup{M,A<:AbstractGroupOperation} <: AbstractGroupManifold{ℝ,A}
    manifold::M
    op::A
end
activate_traits(::DefaultTransparencyGroup) = merge_traits(IsEmbeddedManifold())
