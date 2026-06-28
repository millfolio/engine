"""The `chat` package: prompt rendering + tool-call parsing.

`__init__` re-exports the chat-template surface (from the `chat` submodule) so
`from chat import render_chat` keeps resolving as before. The Gemma helpers and
the tool-call parser live in sibling submodules (`gemma_chat`, `gemma_tools`,
`toolcall`), each re-exported by a top-level shim of the same name."""

from chat.chat import (
    json_escape_str,
    load_chat_template,
    render_value,
    render_request,
    render_chat,
)
