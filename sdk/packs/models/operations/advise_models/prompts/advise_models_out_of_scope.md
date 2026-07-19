# Models Update Error: Out of Scope

The edited manifest touched something other than `mode_models` or `roles`. `/models update` can only reassign which model each mode uses (`mode_models`) and which model each cross-cutting role uses (`roles`, e.g. `deep`) — it cannot add or remove models, reorder the list, or edit per-model fields like `provider`, `cost_tier`, `capabilities`, etc.

```plaintext
{scope_error}
```

Refine the instruction so it only asks for mode/role changes, or edit `models.yaml` directly and run `/models reload`.