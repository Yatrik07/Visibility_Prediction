import sqlite3

# conn = sqlite3.connect('Test.db')
# cursor_obj = conn.cursor()


def createTable():
    conn = sqlite3.connect('Database/Test.db')
    cursor_obj = conn.cursor()
    cursor_obj.execute('''CREATE TABLE if not exists TESTED(DRYBULBTEMPF FLOAT, RelativeHumidity FLOAT,WindSpeed 
    FLOAT,WindDirection FLOAT, SeaLevelPressure FLOAT, Precip FLOAT, PredictedVisibility FLOAT)''');
    print("Table created successfully!")
    conn.close()


# cursor_obj.execute(" select * from TESTED")
# op = cursor_obj.fetchall()

# print(op)


def enterTable(DRYBULBTEMPF, RelativeHumidity, WindSpeed, WindDirection, SeaLevelPressure, Precip , output):
    conn = sqlite3.connect('Database/Test.db')
    cursor_obj = conn.cursor()
    cursor_obj.execute(
        # f'insert into TESTED(DRYBULBTEMPF , RelativeHumidity ,WindSpeed ,WindDirection , SeaLevelPressure ,Precip , PredictedVisibility) values({DRYBULBTEMPF}  , {RelativeHumidity} , {WindSpeed} , {WindDirection} , {SeaLevelPressure}  ,{Precip}, {output} )')
        "insert into TESTED(DRYBULBTEMPF , RelativeHumidity ,WindSpeed ,WindDirection , SeaLevelPressure ,Precip , PredictedVisibility) values(?,?,?,?,?,?,? )",[DRYBULBTEMPF  , RelativeHumidity , WindSpeed , WindDirection , SeaLevelPressure  ,Precip, output])
    conn.commit()
    conn.close()

def showTable():
    conn = sqlite3.connect('../Database/Test.db')
    cursor_obj = conn.cursor()
    output = cursor_obj.execute("select * from TESTED")

    data = cursor_obj.execute('''SELECT * FROM TESTED''')
    for column in data.description:
        print(column[0] , end="\t")
    print("")
    for row in output:
        print(row)
    conn.close()

def dropTabel():
    conn = sqlite3.connect('Test.db')
    cursor_obj = conn.cursor()
    print('hi1')
    cursor_obj.execute('''drop table if exists TESTED''')
    print('hi2')
    conn.commit()
    conn.close()
    print("talbe dropped")

def tables():
    conn = sqlite3.connect('Test.db')
    cursor_obj = conn.cursor()
    cursor_obj.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor_obj.fetchall())

# dropTabel()
# showTable()
# createTable()
# tables()

