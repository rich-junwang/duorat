function(prefix) {
  local databases = [
    'world_1',
  ],

  name: 'spider',
  paths: [
    prefix + 'database/%s/examples_with_synthetic_data_v5_mono_nl_by_t5_gen_full.json' % [db]
    for db in databases
  ],
  tables_paths: [
    prefix + 'database/%s/tables.json' % [db]
    for db in databases
  ],
  db_path: prefix + 'database',
}