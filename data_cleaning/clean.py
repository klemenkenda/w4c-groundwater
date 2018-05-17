import os.path
import pickle
from datetime import date, datetime
from typing import List, Tuple, Any, Iterable, Iterator
import csv

folder: str = os.path.join("data", "raw")

files: List[str] = [os.path.join(folder, file)
                    for file in os.listdir(folder)
                    if file.endswith(".csv")]

files.sort()

FILTER_HEADER = "Nivo v absolutnih kotah (m n.m.)"
allowed_headers = [FILTER_HEADER, "nivo v absolut. kotah (m n.m.)"]

full_data = []  # type:  List[List[Tuple[date, float]]]

special_files = ["85073.csv", "85075.csv", "85076.csv"]


def to_date(s: str) -> date:
    return datetime.strptime(s, "%d.%m.%Y").date()


def parse_row(row: Any, ind: int) -> Tuple[date, float]:
    fst = to_date(row[0])
    if len(row) == 1 or ind >= len(row):
        return fst, float("Nan")
    return fst, float(row[ind])


def parse_special(reader: Iterator[Any]) -> List[Tuple[date, float]]:
    data = []  # type: List[Tuple[date, float]]
    data2 = []  # type: List[Tuple[date, float]]

    for row in reader:
        fst = to_date(row[0])
        if len(row) == 1:
            data.append((fst, float("Nan")))
        if len(row) == 2:
            data.append((fst, float(row[1])))
        if len(row) == 3:
            base = float(row[2])
            data2 = [(fst, base + j / 100) for fst, j in data]
            data2.append((fst, base))
            break

    for row in reader:
        fst = to_date(row[0])
        if len(row) < 3:
            base = float("Nan")
        else:
            base = float(row[2])
        data2.append((fst, base))

    return data2


filenames = []
for i, file in enumerate(files):
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        header = next(reader)  # type: List[str]
        if len(header) == 2:
            ind = 1
            assert header[-1] in allowed_headers
        elif len(header) == 3:
            ind = header.index(FILTER_HEADER)
        elif len(header) == 4:
            ind = 3  # Just for 85024.csv
            if file.endswith("85024.csv"):
                ind = 2
            if any(file.endswith(j) for j in special_files):
                print(f"PARSING SPECIAL: {file}")
                full_data.append(parse_special(reader))
                filenames.append(file)
                continue
            assert file.endswith("85024.csv")
            # Just remove 85024 because it is very unspaced
            continue
        else:
            raise AssertionError("Invalid format")
        data = []  # type: List[Tuple[date, float]]
        for row in reader:
            data.append(parse_row(row, ind))
        filenames.append(file)
        full_data.append(data)


assert len(filenames) == len(full_data)

min_date = min(data[0][0] for data in full_data)
max_date = max(data[-1][0] for data in full_data)

pickle.dump(full_data, open(os.path.join("data", "parsed.pickle"), "wb"))
pickle.dump(filenames, open(os.path.join("data", "filenames.picjle"), "wb"))
print(sum(map(len, full_data)))

print(min_date)
print(max_date)
