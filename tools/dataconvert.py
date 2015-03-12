import re
import csv

file = open("results3.txt")

data = file.read()
lines = data.split("\n")
finalData = []

for i in lines:
  if i[:5] == "Total":
    continue
  finalDataLine = []
  fields = i.split(",")
  for j in fields:
    print j
    try:
      num = float(re.sub("[A-Za-z%()\s':]", "", j))
    except:
      continue
    finalDataLine.append(num)

  finalData.append(finalDataLine)

with open("output.csv", "wb") as f:
  writer = csv.writer(f)
  writer.writerows(finalData)
