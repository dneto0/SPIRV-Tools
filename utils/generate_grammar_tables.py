#!/usr/bin/env python
# Copyright (c) 2016 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates various info tables from SPIR-V JSON grammar."""

import errno
import json
import os.path
import re

# Prefix for all C variables generated by this script.
PYGEN_VARIABLE_PREFIX = 'pygen_variable'

# Extensions to recognize, but which don't necessarily come from the SPIR-V
# core or KHR grammar files.  Get this list from the SPIR-V registery web page.
# NOTE: Only put things on this list if it is not in those grammar files.
EXTENSIONS_FROM_SPIRV_REGISTRY_AND_NOT_FROM_GRAMMARS = """
SPV_AMD_gcn_shader
SPV_AMD_gpu_shader_half_float
SPV_AMD_gpu_shader_int16
SPV_AMD_shader_trinary_minmax
SPV_KHR_non_semantic_info
"""


def make_path_to_file(f):
    """Makes all ancestor directories to the given file, if they don't yet
    exist.

    Arguments:
        f: The file whose ancestor directories are to be created.
    """
    dir = os.path.dirname(os.path.abspath(f))
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            raise


def convert_min_required_version(version):
    """Converts the minimal required SPIR-V version encoded in the grammar to
    the symbol in SPIRV-Tools."""
    if version is None:
        return 'SPV_SPIRV_VERSION_WORD(1, 0)'
    if version == 'None':
        return '0xffffffffu'
    return 'SPV_SPIRV_VERSION_WORD({})'.format(version.replace('.', ','))


def convert_max_required_version(version):
    """Converts the maximum required SPIR-V version encoded in the grammar to
    the symbol in SPIRV-Tools."""
    if version is None:
        return '0xffffffffu'
    return 'SPV_SPIRV_VERSION_WORD({})'.format(version.replace('.', ','))


def compose_capability_list(caps):
    """Returns a string containing a braced list of capabilities as enums.

    Arguments:
      - caps: a sequence of capability names

    Returns:
      a string containing the braced list of SpvCapability* enums named by caps.
    """
    return '{' + ', '.join(['SpvCapability{}'.format(c) for c in caps]) + '}'


def get_capability_array_name(caps):
    """Returns the name of the array containing all the given capabilities.

    Args:
      - caps: a sequence of capability names
    """
    if not caps:
        return 'nullptr'
    return '{}_caps_{}'.format(PYGEN_VARIABLE_PREFIX, ''.join(caps))


def generate_capability_arrays(caps):
    """Returns the arrays of capabilities.

    Arguments:
      - caps: a sequence of sequence of capability names
    """
    caps = sorted(set([tuple(c) for c in caps if c]))
    arrays = [
        'static const SpvCapability {}[] = {};'.format(
            get_capability_array_name(c), compose_capability_list(c))
        for c in caps]
    return '\n'.join(arrays)


def compose_extension_list(exts):
    """Returns a string containing a braced list of extensions as enums.

    Arguments:
      - exts: a sequence of extension names

    Returns:
      a string containing the braced list of extensions named by exts.
    """
    return '{' + ', '.join(
        ['spvtools::Extension::k{}'.format(e) for e in exts]) + '}'


def get_extension_array_name(extensions):
    """Returns the name of the array containing all the given extensions.

    Args:
      - extensions: a sequence of extension names
    """
    if not extensions:
        return 'nullptr'
    else:
        return '{}_exts_{}'.format(
            PYGEN_VARIABLE_PREFIX, ''.join(extensions))


def generate_extension_arrays(extensions):
    """Returns the arrays of extensions.

    Arguments:
      - caps: a sequence of sequence of extension names
    """
    extensions = sorted(set([tuple(e) for e in extensions if e]))
    arrays = [
        'static const spvtools::Extension {}[] = {};'.format(
            get_extension_array_name(e), compose_extension_list(e))
        for e in extensions]
    return '\n'.join(arrays)


def convert_operand_kind(operand_tuple):
    """Returns the corresponding operand type used in spirv-tools for the given
    operand kind and quantifier used in the JSON grammar.

    Arguments:
      - operand_tuple: a tuple of two elements:
          - operand kind: used in the JSON grammar
          - quantifier: '', '?', or '*'

    Returns:
      a string of the enumerant name in spv_operand_type_t
    """
    kind, quantifier = operand_tuple
    # The following cases are where we differ between the JSON grammar and
    # spirv-tools.
    if kind == 'IdResultType':
        kind = 'TypeId'
    elif kind == 'IdResult':
        kind = 'ResultId'
    elif kind == 'IdMemorySemantics' or kind == 'MemorySemantics':
        kind = 'MemorySemanticsId'
    elif kind == 'IdScope' or kind == 'Scope':
        kind = 'ScopeId'
    elif kind == 'IdRef':
        kind = 'Id'

    elif kind == 'ImageOperands':
        kind = 'Image'
    elif kind == 'Dim':
        kind = 'Dimensionality'
    elif kind == 'ImageFormat':
        kind = 'SamplerImageFormat'
    elif kind == 'KernelEnqueueFlags':
        kind = 'KernelEnqFlags'

    elif kind == 'LiteralExtInstInteger':
        kind = 'ExtensionInstructionNumber'
    elif kind == 'LiteralSpecConstantOpInteger':
        kind = 'SpecConstantOpNumber'
    elif kind == 'LiteralContextDependentNumber':
        kind = 'TypedLiteralNumber'

    elif kind == 'PairLiteralIntegerIdRef':
        kind = 'LiteralIntegerId'
    elif kind == 'PairIdRefLiteralInteger':
        kind = 'IdLiteralInteger'
    elif kind == 'PairIdRefIdRef':  # Used by OpPhi in the grammar
        kind = 'Id'

    if kind == 'FPRoundingMode':
        kind = 'FpRoundingMode'
    elif kind == 'FPFastMathMode':
        kind = 'FpFastMathMode'

    if quantifier == '?':
        kind = 'Optional{}'.format(kind)
    elif quantifier == '*':
        kind = 'Variable{}'.format(kind)

    return 'SPV_OPERAND_TYPE_{}'.format(
        re.sub(r'([a-z])([A-Z])', r'\1_\2', kind).upper())


class InstInitializer(object):
    """Instances holds a SPIR-V instruction suitable for printing as the
    initializer for spv_opcode_desc_t."""

    def __init__(self, opname, caps, exts, operands, version, lastVersion):
        """Initialization.

        Arguments:
          - opname: opcode name (with the 'Op' prefix)
          - caps: a sequence of capability names required by this opcode
          - exts: a sequence of names of extensions enabling this enumerant
          - operands: a sequence of (operand-kind, operand-quantifier) tuples
          - version: minimal SPIR-V version required for this opcode
          - lastVersion: last version of SPIR-V that includes this opcode
        """

        assert opname.startswith('Op')
        self.opname = opname[2:]  # Remove the "Op" prefix.
        self.num_caps = len(caps)
        self.caps_mask = get_capability_array_name(caps)
        self.num_exts = len(exts)
        self.exts = get_extension_array_name(exts)
        self.operands = [convert_operand_kind(o) for o in operands]

        self.fix_syntax()

        operands = [o[0] for o in operands]
        self.ref_type_id = 'IdResultType' in operands
        self.def_result_id = 'IdResult' in operands

        self.version = convert_min_required_version(version)
        self.lastVersion = convert_max_required_version(lastVersion)

    def fix_syntax(self):
        """Fix an instruction's syntax, adjusting for differences between the
        officially released grammar and how SPIRV-Tools uses the grammar.

        Fixes:
            - ExtInst should not end with SPV_OPERAND_VARIABLE_ID.
            https://github.com/KhronosGroup/SPIRV-Tools/issues/233
        """
        if (self.opname == 'ExtInst'
                and self.operands[-1] == 'SPV_OPERAND_TYPE_VARIABLE_ID'):
            self.operands.pop()

    def __str__(self):
        template = ['{{"{opname}"', 'SpvOp{opname}',
                    '{num_caps}', '{caps_mask}',
                    '{num_operands}', '{{{operands}}}',
                    '{def_result_id}', '{ref_type_id}',
                    '{num_exts}', '{exts}',
                    '{min_version}', '{max_version}}}']
        return ', '.join(template).format(
            opname=self.opname,
            num_caps=self.num_caps,
            caps_mask=self.caps_mask,
            num_operands=len(self.operands),
            operands=', '.join(self.operands),
            def_result_id=(1 if self.def_result_id else 0),
            ref_type_id=(1 if self.ref_type_id else 0),
            num_exts=self.num_exts,
            exts=self.exts,
            min_version=self.version,
            max_version=self.lastVersion)


class ExtInstInitializer(object):
    """Instances holds a SPIR-V extended instruction suitable for printing as
    the initializer for spv_ext_inst_desc_t."""

    def __init__(self, opname, opcode, caps, operands):
        """Initialization.

        Arguments:
          - opname: opcode name
          - opcode: enumerant value for this opcode
          - caps: a sequence of capability names required by this opcode
          - operands: a sequence of (operand-kind, operand-quantifier) tuples
        """
        self.opname = opname
        self.opcode = opcode
        self.num_caps = len(caps)
        self.caps_mask = get_capability_array_name(caps)
        self.operands = [convert_operand_kind(o) for o in operands]
        self.operands.append('SPV_OPERAND_TYPE_NONE')

    def __str__(self):
        template = ['{{"{opname}"', '{opcode}', '{num_caps}', '{caps_mask}',
                    '{{{operands}}}}}']
        return ', '.join(template).format(
            opname=self.opname,
            opcode=self.opcode,
            num_caps=self.num_caps,
            caps_mask=self.caps_mask,
            operands=', '.join(self.operands))


def generate_instruction(inst, is_ext_inst):
    """Returns the C initializer for the given SPIR-V instruction.

    Arguments:
      - inst: a dict containing information about a SPIR-V instruction
      - is_ext_inst: a bool indicating whether |inst| is an extended
                     instruction.

    Returns:
      a string containing the C initializer for spv_opcode_desc_t or
      spv_ext_inst_desc_t
    """
    opname = inst.get('opname')
    opcode = inst.get('opcode')
    caps = inst.get('capabilities', [])
    exts = inst.get('extensions', [])
    operands = inst.get('operands', {})
    operands = [(o['kind'], o.get('quantifier', '')) for o in operands]
    min_version = inst.get('version', None)
    max_version = inst.get('lastVersion', None)

    assert opname is not None

    if is_ext_inst:
        return str(ExtInstInitializer(opname, opcode, caps, operands))
    else:
        return str(InstInitializer(opname, caps, exts, operands, min_version, max_version))


def generate_instruction_table(inst_table):
    """Returns the info table containing all SPIR-V instructions, sorted by
    opcode, and prefixed by capability arrays.

    Note:
      - the built-in sorted() function is guaranteed to be stable.
        https://docs.python.org/3/library/functions.html#sorted

    Arguments:
      - inst_table: a list containing all SPIR-V instructions.
    """
    inst_table = sorted(inst_table, key=lambda k: (k['opcode'], k['opname']))

    caps_arrays = generate_capability_arrays(
        [inst.get('capabilities', []) for inst in inst_table])
    exts_arrays = generate_extension_arrays(
        [inst.get('extensions', []) for inst in inst_table])

    insts = [generate_instruction(inst, False) for inst in inst_table]
    insts = ['static const spv_opcode_desc_t kOpcodeTableEntries[] = {{\n'
             '  {}\n}};'.format(',\n  '.join(insts))]

    return '{}\n\n{}\n\n{}'.format(caps_arrays, exts_arrays, '\n'.join(insts))


def generate_extended_instruction_table(json_grammar, set_name, operand_kind_prefix=""):
    """Returns the info table containing all SPIR-V extended instructions,
    sorted by opcode, and prefixed by capability arrays.

    Arguments:
      - inst_table: a list containing all SPIR-V instructions.
      - set_name: the name of the extended instruction set.
      - operand_kind_prefix: the prefix, if any, to add to the front
        of operand kind names.
    """
    if operand_kind_prefix:
        prefix_operand_kind_names(operand_kind_prefix, json_grammar)

    inst_table = json_grammar["instructions"]

    inst_table = sorted(inst_table, key=lambda k: k['opcode'])
    caps = [inst.get('capabilities', []) for inst in inst_table]
    caps_arrays = generate_capability_arrays(caps)
    insts = [generate_instruction(inst, True) for inst in inst_table]
    insts = ['static const spv_ext_inst_desc_t {}_entries[] = {{\n'
             '  {}\n}};'.format(set_name, ',\n  '.join(insts))]

    return '{}\n\n{}'.format(caps_arrays, '\n'.join(insts))


class EnumerantInitializer(object):
    """Prints an enumerant as the initializer for spv_operand_desc_t."""

    def __init__(self, enumerant, value, caps, exts, parameters, version, lastVersion):
        """Initialization.

        Arguments:
          - enumerant: enumerant name
          - value: enumerant value
          - caps: a sequence of capability names required by this enumerant
          - exts: a sequence of names of extensions enabling this enumerant
          - parameters: a sequence of (operand-kind, operand-quantifier) tuples
          - version: minimal SPIR-V version required for this opcode
          - lastVersion: last SPIR-V version this opode appears
        """
        self.enumerant = enumerant
        self.value = value
        self.num_caps = len(caps)
        self.caps = get_capability_array_name(caps)
        self.num_exts = len(exts)
        self.exts = get_extension_array_name(exts)
        self.parameters = [convert_operand_kind(p) for p in parameters]
        self.version = convert_min_required_version(version)
        self.lastVersion = convert_max_required_version(lastVersion)

    def __str__(self):
        template = ['{{"{enumerant}"', '{value}', '{num_caps}',
                    '{caps}', '{num_exts}', '{exts}',
                    '{{{parameters}}}', '{min_version}',
                    '{max_version}}}']
        return ', '.join(template).format(
            enumerant=self.enumerant,
            value=self.value,
            num_caps=self.num_caps,
            caps=self.caps,
            num_exts=self.num_exts,
            exts=self.exts,
            parameters=', '.join(self.parameters),
            min_version=self.version,
            max_version=self.lastVersion)


def generate_enum_operand_kind_entry(entry, extension_map):
    """Returns the C initializer for the given operand enum entry.

    Arguments:
      - entry: a dict containing information about an enum entry
      - extension_map: a dict mapping enum value to list of extensions

    Returns:
      a string containing the C initializer for spv_operand_desc_t
    """
    enumerant = entry.get('enumerant')
    value = entry.get('value')
    caps = entry.get('capabilities', [])
    if value in extension_map:
        exts = extension_map[value]
    else:
        exts = []
    params = entry.get('parameters', [])
    params = [p.get('kind') for p in params]
    params = zip(params, [''] * len(params))
    version = entry.get('version', None)
    max_version = entry.get('lastVersion', None)

    assert enumerant is not None
    assert value is not None

    return str(EnumerantInitializer(
        enumerant, value, caps, exts, params, version, max_version))


def generate_enum_operand_kind(enum, synthetic_exts_list):
    """Returns the C definition for the given operand kind.
    It's a static const named array of spv_operand_desc_t.

    Also appends to |synthetic_exts_list| a list of extension lists
    used.
    """
    kind = enum.get('kind')
    assert kind is not None

    # Sort all enumerants according to their values, but otherwise
    # preserve their order so the first name listed in the grammar
    # as the preferred name for disassembly.
    if enum.get('category') == 'ValueEnum':
        def functor(k): return (k['value'])
    else:
        def functor(k): return (int(k['value'], 16))
    entries = sorted(enum.get('enumerants', []), key=functor)

    # SubgroupEqMask and SubgroupEqMaskKHR are the same number with
    # same semantics, but one has no extension list while the other
    # does.  Both should have the extension list.
    # So create a mapping from enum value to the union of the extensions
    # across all those grammar entries.  Preserve order.
    extension_map = {}
    for e in entries:
        value = e.get('value')
        extension_map[value] = []
    for e in entries:
        value = e.get('value')
        exts = e.get('extensions', [])
        for ext in exts:
            if ext not in extension_map[value]:
                extension_map[value].append(ext)
    synthetic_exts_list.extend(extension_map.values())

    name = '{}_{}Entries'.format(PYGEN_VARIABLE_PREFIX, kind)
    entries = ['  {}'.format(generate_enum_operand_kind_entry(e, extension_map))
               for e in entries]

    template = ['static const spv_operand_desc_t {name}[] = {{',
                '{entries}', '}};']
    entries = '\n'.join(template).format(
        name=name,
        entries=',\n'.join(entries))

    return kind, name, entries


def generate_operand_kind_table(enums):
    """Returns the info table containing all SPIR-V operand kinds."""
    # We only need to output info tables for those operand kinds that are enums.
    enums = [e for e in enums if e.get('category') in ['ValueEnum', 'BitEnum']]

    caps = [entry.get('capabilities', [])
            for enum in enums
            for entry in enum.get('enumerants', [])]
    caps_arrays = generate_capability_arrays(caps)

    exts = [entry.get('extensions', [])
            for enum in enums
            for entry in enum.get('enumerants', [])]
    enums = [generate_enum_operand_kind(e, exts) for e in enums]
    exts_arrays = generate_extension_arrays(exts)

    # We have three operand kinds that requires their optional counterpart to
    # exist in the operand info table.
    three_optional_enums = ['ImageOperands', 'AccessQualifier', 'MemoryAccess']
    three_optional_enums = [e for e in enums if e[0] in three_optional_enums]
    enums.extend(three_optional_enums)

    enum_kinds, enum_names, enum_entries = zip(*enums)
    # Mark the last three as optional ones.
    enum_quantifiers = [''] * (len(enums) - 3) + ['?'] * 3
    # And we don't want redefinition of them.
    enum_entries = enum_entries[:-3]
    enum_kinds = [convert_operand_kind(e)
                  for e in zip(enum_kinds, enum_quantifiers)]
    table_entries = zip(enum_kinds, enum_names, enum_names)
    table_entries = ['  {{{}, ARRAY_SIZE({}), {}}}'.format(*e)
                     for e in table_entries]

    template = [
        'static const spv_operand_desc_group_t {p}_OperandInfoTable[] = {{',
        '{enums}', '}};']
    table = '\n'.join(template).format(
        p=PYGEN_VARIABLE_PREFIX, enums=',\n'.join(table_entries))

    return '\n\n'.join((caps_arrays,) + (exts_arrays,) + enum_entries + (table,))


def get_extension_list(instructions, operand_kinds):
    """Returns extensions as an alphabetically sorted list of strings."""

    things_with_an_extensions_field = [item for item in instructions]

    enumerants = sum([item.get('enumerants', [])
                      for item in operand_kinds], [])

    things_with_an_extensions_field.extend(enumerants)

    extensions = sum([item.get('extensions', [])
                      for item in things_with_an_extensions_field
                      if item.get('extensions')], [])

    for item in EXTENSIONS_FROM_SPIRV_REGISTRY_AND_NOT_FROM_GRAMMARS.split():
            # If it's already listed in a grammar, then don't put it in the
            # special exceptions list.
        assert item not in extensions, 'Extension %s is already in a grammar file' % item

    extensions.extend(
        EXTENSIONS_FROM_SPIRV_REGISTRY_AND_NOT_FROM_GRAMMARS.split())

    # Validator would ignore type declaration unique check. Should only be used
    # for legacy autogenerated test files containing multiple instances of the
    # same type declaration, if fixing the test by other methods is too
    # difficult. Shouldn't be used for any other reasons.
    extensions.append('SPV_VALIDATOR_ignore_type_decl_unique')

    return sorted(set(extensions))


def get_capabilities(operand_kinds):
    """Returns capabilities as a list of JSON objects, in order of
    appearance."""
    enumerants = sum([item.get('enumerants', []) for item in operand_kinds
                      if item.get('kind') in ['Capability']], [])
    return enumerants


def generate_extension_enum(extensions):
    """Returns enumeration containing extensions declared in the grammar."""
    return ',\n'.join(['k' + extension for extension in extensions])


def generate_extension_to_string_mapping(extensions):
    """Returns mapping function from extensions to corresponding strings."""
    function = 'const char* ExtensionToString(Extension extension) {\n'
    function += '  switch (extension) {\n'
    template = '    case Extension::k{extension}:\n' \
        '      return "{extension}";\n'
    function += ''.join([template.format(extension=extension)
                         for extension in extensions])
    function += '  };\n\n  return "";\n}'
    return function


def generate_string_to_extension_mapping(extensions):
    """Returns mapping function from strings to corresponding extensions."""

    function = '''
    bool GetExtensionFromString(const char* str, Extension* extension) {{
        static const char* known_ext_strs[] = {{ {strs} }};
        static const Extension known_ext_ids[] = {{ {ids} }};
        const auto b = std::begin(known_ext_strs);
        const auto e = std::end(known_ext_strs);
        const auto found = std::equal_range(
            b, e, str, [](const char* str1, const char* str2) {{
                return std::strcmp(str1, str2) < 0;
            }});
        if (found.first == e || found.first == found.second) return false;

        *extension = known_ext_ids[found.first - b];
        return true;
    }}
    '''.format(strs=', '.join(['"{}"'.format(e) for e in extensions]),
               ids=', '.join(['Extension::k{}'.format(e) for e in extensions]))

    return function


def generate_capability_to_string_mapping(operand_kinds):
    """Returns mapping function from capabilities to corresponding strings.

    We take care to avoid emitting duplicate values.
    """
    function = 'const char* CapabilityToString(SpvCapability capability) {\n'
    function += '  switch (capability) {\n'
    template = '    case SpvCapability{capability}:\n' \
        '      return "{capability}";\n'
    emitted = set()  # The values of capabilities we already have emitted
    for capability in get_capabilities(operand_kinds):
        value = capability.get('value')
        if value not in emitted:
            emitted.add(value)
            function += template.format(capability=capability.get('enumerant'))
    function += '    case SpvCapabilityMax:\n' \
        '      assert(0 && "Attempting to convert SpvCapabilityMax to string");\n' \
        '      return "";\n'
    function += '  };\n\n  return "";\n}'
    return function


def generate_all_string_enum_mappings(extensions, operand_kinds):
    """Returns all string-to-enum / enum-to-string mapping tables."""
    tables = []
    tables.append(generate_extension_to_string_mapping(extensions))
    tables.append(generate_string_to_extension_mapping(extensions))
    tables.append(generate_capability_to_string_mapping(operand_kinds))
    return '\n\n'.join(tables)


def precondition_operand_kinds(operand_kinds):
    """For operand kinds that have the same number, make sure they all have the
    same extension list."""

    # Map operand kind and value to list of the union of extensions
    # for same-valued enumerants.
    exts = {}
    for kind_entry in operand_kinds:
        kind = kind_entry.get('kind')
        for enum_entry in kind_entry.get('enumerants', []):
            value = enum_entry.get('value')
            key = kind + '.' + str(value)
            if key in exts:
                exts[key].extend(enum_entry.get('extensions', []))
            else:
                exts[key] = enum_entry.get('extensions', [])
            exts[key] = sorted(set(exts[key]))

    # Now make each entry the same list.
    for kind_entry in operand_kinds:
        kind = kind_entry.get('kind')
        for enum_entry in kind_entry.get('enumerants', []):
            value = enum_entry.get('value')
            key = kind + '.' + str(value)
            if len(exts[key]) > 0:
                enum_entry['extensions'] = exts[key]

    return operand_kinds


def prefix_operand_kind_names(prefix, json_dict):
    """Modifies json_dict, by prefixing all the operand kind names
    with the given prefix.  Also modifies their uses in the instructions
    to match.
    """

    old_to_new = {}
    for operand_kind in json_dict["operand_kinds"]:
        old_name = operand_kind["kind"]
        new_name = prefix + old_name
        operand_kind["kind"] = new_name
        old_to_new[old_name] = new_name

    for instruction in json_dict["instructions"]:
        for operand in instruction.get("operands", []):
            replacement = old_to_new.get(operand["kind"])
            if replacement is not None:
                operand["kind"] = replacement


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate SPIR-V info tables')

    parser.add_argument('--spirv-core-grammar', metavar='<path>',
                        type=str, required=False,
                        help='input JSON grammar file for core SPIR-V '
                        'instructions')
    parser.add_argument('--extinst-debuginfo-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for DebugInfo extended '
                        'instruction set')
    parser.add_argument('--extinst-cldebuginfo100-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for OpenCL.DebugInfo.100 '
                        'extended instruction set')
    parser.add_argument('--extinst-glsl-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for GLSL extended '
                        'instruction set')
    parser.add_argument('--extinst-opencl-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for OpenCL extended '
                        'instruction set')

    parser.add_argument('--core-insts-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for core SPIR-V instructions')
    parser.add_argument('--glsl-insts-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for GLSL extended instruction set')
    parser.add_argument('--opencl-insts-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for OpenCL extended instruction set')
    parser.add_argument('--operand-kinds-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for operand kinds')
    parser.add_argument('--extension-enum-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for extension enumeration')
    parser.add_argument('--enum-string-mapping-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for enum-string mappings')
    parser.add_argument('--extinst-vendor-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for vendor extended '
                        'instruction set'),
    parser.add_argument('--vendor-insts-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for vendor extended instruction set')
    parser.add_argument('--vendor-operand-kind-prefix', metavar='<string>',
                        type=str, required=False, default=None,
                        help='prefix for operand kinds (to disambiguate operand type enums)')
    args = parser.parse_args()

    if (args.core_insts_output is None) != \
            (args.operand_kinds_output is None):
        print('error: --core-insts-output and --operand-kinds-output '
              'should be specified together.')
        exit(1)
    if args.operand_kinds_output and not (args.spirv_core_grammar and
         args.extinst_debuginfo_grammar and
         args.extinst_cldebuginfo100_grammar):
        print('error: --operand-kinds-output requires --spirv-core-grammar '
              'and --exinst-debuginfo-grammar '
              'and --exinst-cldebuginfo100-grammar')
        exit(1)
    if (args.glsl_insts_output is None) != \
            (args.extinst_glsl_grammar is None):
        print('error: --glsl-insts-output and --extinst-glsl-grammar '
              'should be specified together.')
        exit(1)
    if (args.opencl_insts_output is None) != \
            (args.extinst_opencl_grammar is None):
        print('error: --opencl-insts-output and --extinst-opencl-grammar '
              'should be specified together.')
        exit(1)
    if (args.vendor_insts_output is None) != \
            (args.extinst_vendor_grammar is None):
        print('error: --vendor-insts-output and '
              '--extinst-vendor-grammar should be specified together.')
        exit(1)
    if all([args.core_insts_output is None,
            args.glsl_insts_output is None,
            args.opencl_insts_output is None,
            args.vendor_insts_output is None,
            args.extension_enum_output is None,
            args.enum_string_mapping_output is None]):
        print('error: at least one output should be specified.')
        exit(1)

    if args.spirv_core_grammar is not None:
        with open(args.spirv_core_grammar) as json_file:
            core_grammar = json.loads(json_file.read())
            with open(args.extinst_debuginfo_grammar) as debuginfo_json_file:
                debuginfo_grammar = json.loads(debuginfo_json_file.read())
                with open(args.extinst_cldebuginfo100_grammar) as cldebuginfo100_json_file:
                    cldebuginfo100_grammar = json.loads(cldebuginfo100_json_file.read())
                    prefix_operand_kind_names("CLDEBUG100_", cldebuginfo100_grammar)
                    instructions = []
                    instructions.extend(core_grammar['instructions'])
                    instructions.extend(debuginfo_grammar['instructions'])
                    instructions.extend(cldebuginfo100_grammar['instructions'])
                    operand_kinds = []
                    operand_kinds.extend(core_grammar['operand_kinds'])
                    operand_kinds.extend(debuginfo_grammar['operand_kinds'])
                    operand_kinds.extend(cldebuginfo100_grammar['operand_kinds'])
                    extensions = get_extension_list(instructions, operand_kinds)
                    operand_kinds = precondition_operand_kinds(operand_kinds)
        if args.core_insts_output is not None:
            make_path_to_file(args.core_insts_output)
            make_path_to_file(args.operand_kinds_output)
            with open(args.core_insts_output, 'w') as f:
                f.write(generate_instruction_table(
                    core_grammar['instructions']))
            with open(args.operand_kinds_output, 'w') as f:
                f.write(generate_operand_kind_table(operand_kinds))
        if args.extension_enum_output is not None:
            make_path_to_file(args.extension_enum_output)
            with open(args.extension_enum_output, 'w') as f:
                f.write(generate_extension_enum(extensions))
        if args.enum_string_mapping_output is not None:
            make_path_to_file(args.enum_string_mapping_output)
            with open(args.enum_string_mapping_output, 'w') as f:
                f.write(generate_all_string_enum_mappings(
                    extensions, operand_kinds))

    if args.extinst_glsl_grammar is not None:
        with open(args.extinst_glsl_grammar) as json_file:
            grammar = json.loads(json_file.read())
            make_path_to_file(args.glsl_insts_output)
            with open(args.glsl_insts_output, 'w') as f:
                f.write(generate_extended_instruction_table(
                    grammar, 'glsl'))

    if args.extinst_opencl_grammar is not None:
        with open(args.extinst_opencl_grammar) as json_file:
            grammar = json.loads(json_file.read())
            make_path_to_file(args.opencl_insts_output)
            with open(args.opencl_insts_output, 'w') as f:
                f.write(generate_extended_instruction_table(
                    grammar, 'opencl'))

    if args.extinst_vendor_grammar is not None:
        with open(args.extinst_vendor_grammar) as json_file:
            grammar = json.loads(json_file.read())
            make_path_to_file(args.vendor_insts_output)
            name = args.extinst_vendor_grammar
            start = name.find('extinst.') + len('extinst.')
            name = name[start:-len('.grammar.json')].replace('-', '_')
            with open(args.vendor_insts_output, 'w') as f:
                f.write(generate_extended_instruction_table(
                    grammar, name, args.vendor_operand_kind_prefix))


if __name__ == '__main__':
    main()
