# MySQL Importer

Imports local daily K-line CSV files from `data/daily` into the cloud MySQL table `emstocks.mkandles`.

Required environment variable:

- `MYSQL_PASSWORD`

Optional environment variables:

- `MYSQL_HOST`
- `MYSQL_PORT` default `3306`
- `MYSQL_USER`
- `MYSQL_DATABASE`
- `MKANDLES_KTYPE` default `M`

Run:

```powershell
$env:MYSQL_PASSWORD='your-password'
python .\mysql_importer\import_daily_to_mysql.py
```

The script deletes existing `mkandles` rows by default before importing. Add `--no-clear` to append instead.