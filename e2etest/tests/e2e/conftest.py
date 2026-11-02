# e2etest keeps pytest away from scenario assets; this dir is driven by the
# lisa-e2e CLI, not collected as tests.
collect_ignore_glob = ["fixtures/*", "model/*", "*.scenario.yml"]
