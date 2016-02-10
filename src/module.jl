# CUDA module management

immutable CuModule
    handle::Ptr{Void}

    "Create a module with kernel functions from given PTX module"
    function CuModule(filename::ASCIIString)
        a = Array(Ptr{Void}, 1)
        @cucall(cuModuleLoad, (Ptr{Ptr{Void}}, Ptr{Cchar}), a, filename)
        new(a[1])
    end
end

"Unload the module once used"
function unload(md::CuModule)
    @cucall(cuModuleUnload, (Ptr{Void},), md.handle)
end


immutable CuFunction
    handle::Ptr{Void}

    "Load a kernel function from the module"
    function CuFunction(md::CuModule, name::ASCIIString)
        a = Array(Ptr{Void}, 1)
        @cucall(cuModuleGetFunction, (Ptr{Ptr{Void}}, Ptr{Void}, Ptr{Cchar}),
            a, md.handle, name)
        new(a[1])
    end
end


