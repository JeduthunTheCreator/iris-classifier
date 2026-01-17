temperatures = [22.5, 19.0, 27.3, 21.8, 23.4]
threshold = 22.0
count = 0
above_the_threshold = []

for temp in temperatures:
    if temp > threshold:
        count += 1
        above_the_threshold.append(temp)

print(f"{count} temperatures are above {threshold}°C: {above_the_threshold}")
