"""Gemma 4 tool-declaration formatter (the system-block <|tool>…<tool|> blocks).

The upstream gemma4 chat template renders tool definitions with recursive Jinja
macros (format_function_declaration / format_parameters / format_argument) using
dictsort + | upper + is-mapping tests — none of which jinja2.mojo supports. So
the block is built here in Mojo, byte-for-byte identical to the macros, and
injected into the template as a pre-rendered `gemma_tools_block` string.

Format (validated against transformers.apply_chat_template):
  <|tool>declaration:NAME{description:<|"|>DESC<|"|>,parameters:{
     properties:{KEY:{description:<|"|>…<|"|>,<type-specific>,type:<|"|>TYPE<|"|>},…},
     required:[<|"|>…<|"|>,…],type:<|"|>OBJECT<|"|>}}<tool|>
Properties are sorted by key (dictsort); types are upper-cased; `<|"|>` is the
quote token. Covers string+enum, number, boolean, array(items), nested object,
nullable, required — the macro branches.
"""

from value import Value, VNONE, VBOOL, VINT, VFLOAT, VSTR, VLIST, VMAP

comptime Q = '<|"|>'  # the gemma quote control token


def _upper(s: String) -> String:
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        var c = Int(b[i])
        out.append(UInt8(c - 32) if (c >= 97 and c <= 122) else b[i])
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _less(a: String, b: String) -> Bool:
    """Byte-wise string ordering (dictsort key order)."""
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var n = len(ab) if len(ab) < len(bb) else len(bb)
    for i in range(n):
        if ab[i] != bb[i]:
            return ab[i] < bb[i]
    return len(ab) < len(bb)


def dictsort(v: Value) -> List[Int]:
    """Indices into v's parallel keys/vals, sorted alphabetically by key."""
    var idx = List[Int]()
    for i in range(len(v.c[].keys)):
        idx.append(i)
    # insertion sort (tool property counts are tiny)
    for i in range(1, len(idx)):
        var j = i
        while j > 0 and _less(v.c[].keys[idx[j]], v.c[].keys[idx[j - 1]]):
            var t = idx[j]
            idx[j] = idx[j - 1]
            idx[j - 1] = t
            j -= 1
    return idx^


def _truthy(v: Value) -> Bool:
    if v.tag == VNONE:
        return False
    if v.tag == VBOOL:
        return v.b
    if v.tag == VSTR:
        return v.s.byte_length() > 0
    if v.tag == VINT:
        return v.i != 0
    if v.tag == VFLOAT:
        return v.f != 0.0
    if v.tag == VLIST:
        return len(v.c[].vals) > 0
    if v.tag == VMAP:
        return len(v.c[].keys) > 0
    return False  # VUNDEF


def _get(v: Value, key: String) -> Value:
    """v[key] or a none Value if absent."""
    var o = v.map_get(key)
    if o:
        return o.value()
    return Value.none()


def _scalar(v: Value) -> String:
    if v.tag == VINT:
        return String(v.i)
    if v.tag == VFLOAT:
        return String(v.f)
    if v.tag == VBOOL:
        return String("true") if v.b else String("false")
    if v.tag == VSTR:
        return v.s.copy()
    return String("")


def _str_list(v: Value) -> String:
    """`<|"|>a<|"|>,<|"|>b<|"|>` for a list of strings (required[]/enum-as-required).
    """
    var out = String("")
    for i in range(len(v.c[].vals)):
        if i > 0:
            out += ","
        out += Q + v.c[].vals[i].s + Q
    return out^


def format_argument(arg: Value, escape_keys: Bool) raises -> String:
    if arg.tag == VSTR:
        return Q + arg.s + Q
    if arg.tag == VBOOL:
        return String("true") if arg.b else String("false")
    if arg.tag == VMAP:
        var out = String("{")
        var first = True
        var idx = dictsort(arg)
        for ii in range(len(idx)):
            if not first:
                out += ","
            first = False
            var key = arg.c[].keys[idx[ii]]
            out += (Q + key + Q) if escape_keys else key
            out += ":" + format_argument(arg.c[].vals[idx[ii]], escape_keys)
        out += "}"
        return out^
    if arg.tag == VLIST:
        var out = String("[")
        for i in range(len(arg.c[].vals)):
            if i > 0:
                out += ","
            out += format_argument(arg.c[].vals[i], escape_keys)
        out += "]"
        return out^
    return _scalar(arg)


comptime _STD_KEYS = "description type properties required nullable"


def _is_standard(key: String) -> Bool:
    return (
        key == "description"
        or key == "type"
        or key == "properties"
        or key == "required"
        or key == "nullable"
    )


def _format_items(items: Value) raises -> String:
    """The ARRAY `items:{…}` body (dictsort over item keys, type-special)."""
    var out = String("")
    var first = True
    var idx = dictsort(items)
    for ii in range(len(idx)):
        var key = items.c[].keys[idx[ii]]
        var val = items.c[].vals[idx[ii]]
        if val.tag == VNONE:
            continue
        if not first:
            out += ","
        first = False
        if key == "properties":
            out += "properties:{" + format_parameters(val, False) + "}"
        elif key == "required":
            out += "required:[" + _str_list(val) + "]"
        elif key == "type":
            if val.tag == VSTR:
                out += "type:" + Q + _upper(val.s) + Q
            else:  # list of types -> [<|"|>A<|"|>,…] uppercased
                var ups = Value.list_of(List[Value]())
                for t in range(len(val.c[].vals)):
                    ups.c[].vals.append(Value.string(_upper(val.c[].vals[t].s)))
                out += "type:" + format_argument(ups, True)
        else:
            out += key + ":" + format_argument(val, True)
    return out^


def _property_body(value: Value) raises -> String:
    """Inner body of one property `{…}` incl. the trailing `}` (after type)."""
    var out = String("")
    var add_comma = False
    var vtype = _upper(_get(value, "type").s)

    var desc = _get(value, "description")
    if _truthy(desc):
        out += "description:" + Q + desc.s + Q
        add_comma = True

    if vtype == "STRING":
        var en = _get(value, "enum")
        if _truthy(en):
            if add_comma:
                out += ","
            add_comma = True
            out += "enum:" + format_argument(en, True)
    elif vtype == "ARRAY":
        var items = _get(value, "items")
        if items.tag == VMAP and _truthy(items):
            if add_comma:
                out += ","
            add_comma = True
            out += "items:{" + _format_items(items) + "}"

    if _truthy(_get(value, "nullable")):
        if add_comma:
            out += ","
        add_comma = True
        out += "nullable:true"

    if vtype == "OBJECT":
        var props = _get(value, "properties")
        if props.tag == VMAP:
            if add_comma:
                out += ","
            add_comma = True
            out += "properties:{" + format_parameters(props, False) + "}"
        elif value.tag == VMAP:
            if add_comma:
                out += ","
            add_comma = True
            out += "properties:{" + format_parameters(value, True) + "}"
        var req = _get(value, "required")
        if _truthy(req):
            if add_comma:
                out += ","
            add_comma = True
            out += "required:[" + _str_list(req) + "]"

    if add_comma:
        out += ","
    out += "type:" + Q + vtype + Q + "}"
    return out^


def format_parameters(properties: Value, filter_keys: Bool) raises -> String:
    var out = String("")
    var found_first = False
    var idx = dictsort(properties)
    for ii in range(len(idx)):
        var key = properties.c[].keys[idx[ii]]
        if filter_keys and _is_standard(key):
            continue
        if found_first:
            out += ","
        found_first = True
        out += key + ":{" + _property_body(properties.c[].vals[idx[ii]])
    return out^


def format_function_declaration(tool: Value) raises -> String:
    var f = _get(tool, "function")
    var name = _get(f, "name").s
    var desc = _get(f, "description").s
    var out = String("declaration:") + name + "{description:" + Q + desc + Q
    var params = _get(f, "parameters")
    if _truthy(params):
        out += ",parameters:{"
        var props = _get(params, "properties")
        if _truthy(props):
            out += "properties:{" + format_parameters(props, False) + "},"
        var req = _get(params, "required")
        if _truthy(req):
            out += "required:[" + _str_list(req) + "],"
        var ptype = _get(params, "type")
        if _truthy(ptype):
            out += "type:" + Q + _upper(ptype.s) + Q + "}"
    out += "}"
    return out^


def format_gemma_tools(tools: Value) raises -> String:
    """The full system-block tool segment: each tool wrapped <|tool>…<tool|>."""
    var out = String("")
    for i in range(len(tools.c[].vals)):
        out += (
            "<|tool>"
            + format_function_declaration(tools.c[].vals[i])
            + "<tool|>"
        )
    return out^
