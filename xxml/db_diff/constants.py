MYSQLDUMP_CMD = 'mysqldump -h {host} -u {user} -p{password} --skip-comments --skip-extended-insert --skip-data --skip-lock-tables {db} > {backup_file}'