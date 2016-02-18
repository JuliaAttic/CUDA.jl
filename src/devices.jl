# CUDA CuDevice management

"Return count of number of GPUs on machine"
function devcount()
    # Get the number of CUDA-capable CuDevices
    a = Ref{Cint}()
    @cucall(cuDeviceGetCount, (Ptr{Cint},), a)
    return Int(a[])
end


immutable CuDevice
    ordinal::Cint
    handle::Cint

    function CuDevice(i::Int)
        ordinal = convert(Cint, i)
        a = Ref{Cint}()
        @cucall(cuDeviceGet, (Ptr{Cint}, Cint), a, ordinal)
        handle = a[]
        new(ordinal, handle)
    end
end

immutable CuCapability
    major::Int
    minor::Int
end

"Get name of a particular device"
function name(dev::CuDevice)
    const buflen = 256
    buf = Array(Cchar, buflen)
    @cucall(cuDeviceGetName, (Ptr{Cchar}, Cint, Cint), buf, buflen, dev.handle)
    bytestring(pointer(buf))
end

"Get total GPU memory of a device in bytes"
function totalmem(dev::CuDevice)
    a = Ref{Csize_t}()
    @cucall(cuDeviceTotalMem, (Ptr{Csize_t}, Cint), a, dev.handle)
    return Int(a[])
end

function attribute(dev::CuDevice, attrcode::Integer)
    a = Ref{Cint}()
    @cucall(cuDeviceGetAttribute, (Ptr{Cint}, Cint, Cint), a, attrcode, dev.handle)
    return Int(a[])
end

"Return the CUDA capability of device as a CuCapability object in the format (majorversion, minorversion)"
capability(dev::CuDevice) = CuCapability(attribute(dev, 75), attribute(dev, 76))

"List all devices with their capabilities and attributes"
function list_devices()
    cnt = devcount()
    if cnt == 0
        println("No CUDA-capable CuDevice found.")
        return
    end

    for i = 0:cnt-1
        dev = CuDevice(i)
        nam = name(dev)
        tmem = iround(totalmem(dev) / (1024^2))
        cap = capability(dev)

        println("device[$i]: $(nam), capability $(cap.major).$(cap.minor), total mem = $tmem MB")
    end
end

