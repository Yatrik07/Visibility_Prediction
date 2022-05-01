import sqlite3

conn = sqlite3.connect('Test.db')
cursor_obj = conn.cursor()


def createTable():
    conn = sqlite3.connect('Test.db')
    cursor_obj = conn.cursor()
    cursor_obj.execute('''
        CREATE TABLE if not exists TESTED(DRYBULBTEMPF FLOAT,	WETBULBTEMPF FLOAT,RelativeHumidity FLOAT,WindSpeed FLOAT,WindDirection FLOAT,SeaLevelPressure FLOAT,Precip FLOAT)''');
    print("Table created successfully!")
    conn.close()


# cursor_obj.execute(" select * from TESTED")
# op = cursor_obj.fetchall()

# print(op)


def enterTable(DRYBULBTEMPF, WETBULBTEMPF, RelativeHumidity, WindSpeed, WindDirection, SeaLevelPressure, Precip):
    conn = sqlite3.connect('Test.db')
    cursor_obj = conn.cursor()
    cursor_obj.execute(f'''insert into TESTED values({DRYBULBTEMPF} , {WETBULBTEMPF} , {RelativeHumidity} , {WindSpeed} , {WindDirection} , {SeaLevelPressure}  ,{Precip} )''')
    conn.commit()
    conn.close()

def showTable():
    conn = sqlite3.connect('Test.db')
    cursor_obj = conn.cursor()
    cursor_obj.execute("select * from TESTED")
    output = cursor_obj.fetchall()
    print(output)

def dropTabel():
    conn = sqlite3.connect('Test.db')
    cursor_obj = conn.cursor()
    print('hi1')
    cursor_obj.execute('''drop table if exists TESTED''')
    print('hi2')
    conn.commit()
    conn.close()

dropTabel()

conn.close()