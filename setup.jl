using Pkg;

function strip_quotes(str)
    if startswith(str, "`") || startswith(str, "\"")
        str = str[2:end]
    end
    if endswith(str, "`") || endswith(str, "\"")
        str = str[1:end-1]
    end
    return str
end

for line in readlines("./REQUIRE")
    package_data = split(line)
    n,v = package_data[1:2]
    println("$(n) - $(v)") 
    if length(package_data) < 3     
        v = startswith(v, "v") ? v[2:end] : v;
        Pkg.add(name=n, version=String(v))
    else
        giturl = strip_quotes(package_data[3])
        Pkg.add(url=giturl)
    end
end
