The bundled base-data dir ships no `.env` file: scripted scenarios need no
credentials. Copy a `.env` file here only when running live scenarios against
your own base data dir (`LISA_E2E_BASE_DATA_DIR`).

## Notes

This directory is the default base-data root for `lisa-e2e`; point
`LISA_E2E_BASE_DATA_DIR` at your own directory to reuse live base data.
