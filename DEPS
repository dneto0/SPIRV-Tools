use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': 'f489ceb630c633fe524ce6e72107e8cc930de14b',

  'effcee_revision': 'e63a164aa0a40a04ddca2c18976819668b5a47a8',

  'googletest_revision': '283c17563fe7a1111cd7f581aa5d541e8baeff2f',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': 'a380cd25433092dbce9a455a3feb1242138febee',
}

deps = {
  'external/abseil_cpp':
      Var('github') + '/abseil/abseil-cpp.git@' + Var('abseil_revision'),

  'external/effcee':
      Var('github') + '/google/effcee.git@' + Var('effcee_revision'),

  'external/googletest':
      Var('github') + '/google/googletest.git@' + Var('googletest_revision'),

  'external/protobuf':
      Var('github') + '/protocolbuffers/protobuf.git@' + Var('protobuf_revision'),

  'external/re2':
      Var('github') + '/google/re2.git@' + Var('re2_revision'),

  'external/spirv-headers':
      Var('github') +  '/KhronosGroup/SPIRV-Headers.git@' +
          Var('spirv_headers_revision'),
}

