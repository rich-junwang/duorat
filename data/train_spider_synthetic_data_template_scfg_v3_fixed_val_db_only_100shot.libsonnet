function(prefix) {
  local databases = [
    'concert_singer',
    'pets_1',
    'car_1',
    'flight_2',
    'employee_hire_evaluation',
    'cre_Doc_Template_Mgt',
    'course_teach',
    'museum_visit',
    'wta_1',
    'battle_death',
    'student_transcripts_tracking',
    'tvshow',
    'poker_player',
    'voter_1',
    'world_1',
    'orchestra',
    'network_1',
    'dog_kennels',
    'singer',
    'real_estate_properties'
  ],

  name: 'spider',
  paths: [
    prefix + 'database/%s/examples_spider_synthetic_data_template_scfg_v3_fixed_val_db_only_100shot.json' % [db]
    for db in databases
  ],
  tables_paths: [
    prefix + 'database/%s/tables.json' % [db]
    for db in databases
  ],
  db_path: prefix + 'database',
}