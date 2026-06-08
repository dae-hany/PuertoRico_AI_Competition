# submissions/

Drop your agent file here to debug it in the web UI.

- Any `.py` file in this folder is scanned by the web UI (`webui/server.py`), and
  every `Agent` subclass it defines appears in each seat's dropdown.
- See [`example_agent.py`](example_agent.py) for a minimal example, or copy
  [`../submission_template/my_agent.py`](../submission_template/my_agent.py).
- Files starting with `_` are ignored.

```bash
python webui/server.py     # then open http://127.0.0.1:5000 and pick your agent
```

You don't have to use this folder — in the web UI you can also type
`module:Class` or `path/to/file.py:Class` directly into a seat's box.
