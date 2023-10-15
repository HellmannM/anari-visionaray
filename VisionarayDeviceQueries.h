// Copyright 2021 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

// This file was generated by generate_queries.py
// Don't make changes to this directly

#include <anari/anari.h>
namespace visionaray {
#define ANARI_INFO_required 0
#define ANARI_INFO_default 1
#define ANARI_INFO_minimum 2
#define ANARI_INFO_maximum 3
#define ANARI_INFO_description 4
#define ANARI_INFO_elementType 5
#define ANARI_INFO_value 6
#define ANARI_INFO_sourceExtension 7
#define ANARI_INFO_extension 8
#define ANARI_INFO_parameter 9
#define ANARI_INFO_channel 10
#define ANARI_INFO_use 11
const int extension_count = 17;
const char ** query_extensions();
const char ** query_object_types(ANARIDataType type);
const ANARIParameter * query_params(ANARIDataType type, const char *subtype);
const void * query_param_info_enum(ANARIDataType type, const char *subtype, const char *paramName, ANARIDataType paramType, int infoName, ANARIDataType infoType);
const void * query_param_info(ANARIDataType type, const char *subtype, const char *paramName, ANARIDataType paramType, const char *infoNameString, ANARIDataType infoType);
const void * query_object_info_enum(ANARIDataType type, const char *subtype, int infoName, ANARIDataType infoType);
const void * query_object_info(ANARIDataType type, const char *subtype, const char *infoNameString, ANARIDataType infoType);
}
