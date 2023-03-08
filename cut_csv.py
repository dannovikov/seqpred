#takes a csv containing a single row of data and returns the first half of the row

csv_path = r"data\USA\usa_int_dates.csv"
output_path = r"data\USA\usa_int_dates_50k.csv"

cut_size = 50000

with open (csv_path, "r") as f:
    data = f.read()
    data = data.split(",")
    data = data[:cut_size]
    print(len(data))
    data = [int(x) for x in data]

    with open (output_path, "w") as f:
        f.write(",".join([str(x) for x in data]))