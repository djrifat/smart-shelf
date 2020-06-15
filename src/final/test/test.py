import csv
csv_columns = ['ID','Gender','Age','Emotion','Dwell time', 'Timestamp']
dict_data = [
    {
        'No': 1, 
        'Name': 'Alex', 
        'Country': 
        'India'
    },
    {
        'No': 2, 
        'Name': 'Ben', 
        'Country': 'USA'
    },
    {
        'No': 3, 
        'Name': 'Shri Ram', 
        'Country': 'India'
    },
    {
        'No': 4, 
        'Name': 'Smith', 
        'Country': 'USA'
        },
    {
        'No': 5, 
        'Name': 'Yuva Raj', 
        'Country': 'Ha'
    },
]


foo = [
    {
        'ID': 1,
        'Gender': 'Male',
        'Age': 20,
        'Emotion': 'Happy',
        'Dwell time': '00:08:45',
        'Timestamp': '18:44:09'
    },
    {
        'ID': 2,
        'Gender': 'Female',
        'Age': 45,
        'Emotion': 'Angry',
        'Dwell time': '00:18:45',
        'Timestamp': '15:44:09'
    },
]

foo_bar = [
    {
        'ID': ID,
        'Gender': gender,
        'Age': age,
        'Emotion': emotion,
        'Dwell time': dwell_time,
        'Timestamp': timestamp
    },
    {
        'ID': ID,
        'Gender': gender,
        'Age': age,
        'Emotion': emotion,
        'Dwell time': dwell_time,
        'Timestamp': timestamp
    },
]


print(type(foo))
# WRITE DATA WITHOUT INDEX
'''
mydict = {
    'key1': 'value_a', 
    'key2': 'value_b', 
    'key3': 'value_c'
}

# For python 2, skip the "newline" argument: open('dict.csv','w")
with open('dict.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in mydict.items():
       writer.writerow([key, value])
'''

csv_file = "csv-files/Names.csv"

# WRITE DICT WITH INDEX, INDEX IS PRE DEFINED IN DICT
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in foo:
            writer.writerow(data)
except IOError:
    print("I/O error")





