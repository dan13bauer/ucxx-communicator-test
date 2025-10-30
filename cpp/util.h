#pragma once

#include <rmm/mr/device/device_memory_resource.hpp>

std::shared_ptr<rmm::mr::device_memory_resource> createMemoryResource(
    std::string_view mode,
    int percent = 10);
