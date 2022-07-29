function(prefix) {
  local databases = [
    'world_1',
  ],

  name: 'spider',
  paths: [
    prefix + 'database/%s/examples.json' % [db]
    for db in databases
  ],
  tables_paths: [
    prefix + 'database/%s/tables.json' % [db]
    for db in databases
  ],
  db_path: prefix + 'database',
}