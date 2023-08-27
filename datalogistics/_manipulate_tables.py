def add_column_to_db(tbl_to_add_to
                     , pandas_from_which_to_add
                     , col_to_add
                     , col_name_within_db
                     , connection
                     , control_col="time"):
    # Create temporary table to add to DB
    tmp = pandas_from_which_to_add.loc[:, [col_to_add, control_col]]
    tmp["time"] = tmp["time"].dt.strftime("%Y-%m-%d")
    # Add to DB
    tmp.to_sql('tmp', connection, if_exists='replace', index=True)
    connection.execute('ALTER TABLE {} ADD COLUMN {} REAL'.format(tbl_to_add_to
                                                                  , col_name_within_db))
    # Add column within DB
    qry = 'UPDATE {} SET {} = (SELECT {} FROM tmp WHERE tmp.{} = {}.{}) where {} is NULL'.format(
    tbl_to_add_to
        , col_name_within_db
        , col_to_add
        , control_col
        , tbl_to_add_to
        , control_col
        , col_name_within_db)
    connection.execute(qry)
    connection.commit()
    # Drop temporary table from DB
    connection.execute('drop table tmp')
