set -e
# input: a file with SQL command in my SQL format
MYSQL_INPUT=$1; shift
# output: an SQLITE database
SQLITE_OUTPUT=$1; shift
ROOT_PATH=$1; shift
set +e

echo DROP DATABASE DB | mysql -uroot
echo CREATE DATABASE DB | mysql -uroot
mysql -uroot -D DB < $MYSQL_INPUT
cd $ROOT_PATH
bash ./mysql2sqlite.sh $@ -uroot DB | sqlite3 $ROOT_PATH/data/database/$SQLITE_OUTPUT
