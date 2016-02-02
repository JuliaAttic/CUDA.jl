module CUDA

    import Base.length, Base.size
    import Base.copy!

    export

    # errors
    CuDriverError, description,

    # base
    @cucall, 

    # devices
    CuDevice, CuCapability, dev_count, name, totalmem, attribute, capability,
    list_devices,

    # context
    CuContext, create_context, destroy, push, pop,

    # module
    CuModule, CuFunction, unload,

    # stream
    CuStream, synchronize,

    # execution
    culaunch,

    # arrays
    CuPtr, CuArray, free, to_host

    const CUDA_LIB = @windows? "nvcuda.dll" : "libcuda"
    Libdl.dlopen(CUDA_LIB)    # loads library, throws an error if not found

    include("errors.jl")
    include("funmap.jl")

    include("base.jl")
    include("devices.jl")
    include("context.jl")
    include("module.jl")
    include("stream.jl")
    include("execution.jl")

    include("arrays.jl")
end
