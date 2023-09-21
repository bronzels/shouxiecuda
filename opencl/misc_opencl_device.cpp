#include <vector>
#include <iostream>

#include <CL/cl.hpp>

#include "../cl_common.h"

int main(int argc, char** argv) {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout<<" No platforms found. Check OpenCL installations!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "OpenCL: Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0) {
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    cl::Device default_device = all_devices[0];
    std::cout << "OpenCL: Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;
}