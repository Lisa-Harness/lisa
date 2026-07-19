# Models Update Error: Out of Scope

The edited manifest touched something other than `mode_models` or `designations`. `/models update` can only reassign which model each mode uses and which models hold the `default` / `deep` / `reasoning` designations — it cannot add or remove models, reorder the list, or edit per-model fields like `provider`, `cost_tier`, `capabilities`, etc.

```plaintext
{scope_error}
```

Refine the instruction so it only asks for mode/designation changes, or edit `models.yaml` directly and run `/models reload`.