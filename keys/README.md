# keys/

Study materials the pipeline needs. None of them are in the repository; the files below are
gitignored and stay on the machine that runs the pipeline.

| File | Content | Made by |
|---|---|---|
| `word_lists.json` | `{"list_1": [12 words], ...}`, the studied words of each list | `src/io_index.py`, from the watch export |
| `word_lists_ordered.json` | `{"list_1": {"memory": [12 words in presentation order], "foils": [...]}, ...}` | supplied by the study; needed for serial position |
| `locked_participants.txt` | study numbers of the held-out participants, one per line or comma separated | the study; see `config.py` |

`word_lists_ordered.example.json` shows the expected format with made-up words.
