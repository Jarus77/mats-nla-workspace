# Golden files

Data the tests compare the port against, byte for byte: option lists, rendered prompts, marker
offsets and regex verdicts. Most are frozen outputs of the source repo's own scripts (Camila's
private `global-workspace`, 2026-09-15..22); the rest (`capable_questions.json` and the option
lists of the families this repo drew itself) pin this repo's own builders. They are inputs to
`tests/test_<family>.py`, never regenerated here.

The scripts that produced the source-repo goldens (`make_<family>.py`, importing that repo by an
absolute path) were removed in the 2026-09-23 audit because nobody outside that machine could
run them; they are in the git history before that commit if a golden ever has to be rebuilt.
